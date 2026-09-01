#include "worker_process.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "protocol.h"

namespace
{
    constexpr std::size_t kMaxValuesPerTask = 1ULL << 26; // 67,108,864 doubles (~536 MiB payload)

    void close_pipe_pair(int (&fds)[2])
    {
        close(fds[0]);
        close(fds[1]);
    }

    void validate_worker_path(const std::string &path, const char *what)
    {
        if (path.empty())
            throw std::invalid_argument(std::string(what) + " must not be empty.");

        if (path.find('/') == std::string::npos && path.find('~') != 0)
            return;

        struct stat st{};
        if (stat(path.c_str(), &st) != 0)
            throw std::invalid_argument(std::string(what) + " does not exist: " + path);
        if (!S_ISREG(st.st_mode))
            throw std::invalid_argument(std::string(what) + " is not a regular file: " + path);
    }

    std::size_t checked_size_product(std::size_t a, std::size_t b)
    {
        if (a == 0 || b == 0)
            return 0;
        if (a > kMaxValuesPerTask / b)
            throw std::invalid_argument("Task payload is too large.");
        return a * b;
    }
} // namespace

// CPU pinning support removed; no platform-specific scheduler headers needed.
class WorkerProcessImpl
{
public:
    WorkerProcessImpl(const std::string &pythonExe, const std::string &scriptPath, std::size_t qrcLayers, std::size_t nJobs);
    ~WorkerProcessImpl();

    WorkerProcessImpl(const WorkerProcessImpl &) = delete;
    WorkerProcessImpl &operator=(const WorkerProcessImpl &) = delete;

    std::vector<double> process_chunk(std::size_t taskId, const DataView &input, std::size_t &resultCols);

private:
    bool write_exact(const void *buf, std::size_t len);
    bool read_exact(void *buf, std::size_t len);

    pid_t childPid_{-1};
    int childInFd_{-1};
    int childOutFd_{-1};
};

WorkerProcessImpl::WorkerProcessImpl(const std::string &pythonExe, const std::string &scriptPath, std::size_t qrcLayers, std::size_t nJobs)
{
    signal(SIGPIPE, SIG_IGN);

    validate_worker_path(pythonExe, "Python executable");
    validate_worker_path(scriptPath, "Worker script");
    if (qrcLayers == 0)
        throw std::invalid_argument("qrc-layers must be > 0.");
    if (nJobs == 0)
        throw std::invalid_argument("n-jobs must be > 0.");

    int toChild[2], fromChild[2];
    if (pipe(toChild) != 0 || pipe(fromChild) != 0)
    {
        throw std::runtime_error("Failed to create pipes for worker process.");
    }

    childPid_ = fork();
    if (childPid_ < 0)
    {
        close_pipe_pair(toChild);
        close_pipe_pair(fromChild);
        throw std::runtime_error("Failed to fork worker process.");
    }

    if (childPid_ == 0)
    {
        dup2(toChild[0], STDIN_FILENO);
        dup2(fromChild[1], STDOUT_FILENO);
        for (int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]})
            close(fd);

        const std::string layersArg = std::to_string(qrcLayers);
        const std::string jobsArg = std::to_string(nJobs);
        char *argv[] = {
            const_cast<char *>(pythonExe.c_str()),
            const_cast<char *>(scriptPath.c_str()),
            const_cast<char *>("--qrc-layers"),
            const_cast<char *>(layersArg.c_str()),
            const_cast<char *>("--n-jobs"),
            const_cast<char *>(jobsArg.c_str()),
            nullptr};

        if (pythonExe.find('/') != std::string::npos)
            execv(pythonExe.c_str(), argv);
        else
            execvp(pythonExe.c_str(), argv);
        _exit(127);
    }

    close(toChild[0]);
    close(fromChild[1]);
    childInFd_ = toChild[1];
    childOutFd_ = fromChild[0];
}

WorkerProcessImpl::~WorkerProcessImpl()
{
    if (childPid_ <= 0)
        return;

    if (childInFd_ >= 0)
    {
        MessageHeader quit{static_cast<uint32_t>(Protocol::Magic), kVersion,
                           static_cast<uint16_t>(MessageType::Quit), 0, 0, 0};
        write_exact(&quit, sizeof(quit));
        close(childInFd_);
    }
    if (childOutFd_ >= 0)
        close(childOutFd_);

    waitpid(childPid_, nullptr, 0);
}

bool WorkerProcessImpl::write_exact(const void *buf, std::size_t len)
{
    if (len == 0)
        return true;

    const auto *p = static_cast<const char *>(buf);
    std::size_t written = 0;
    while (written < len)
    {
        const ssize_t n = write(childInFd_, p + written, len - written);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        written += static_cast<std::size_t>(n);
    }
    return true;
}

bool WorkerProcessImpl::read_exact(void *buf, std::size_t len)
{
    if (len == 0)
        return true;

    auto *p = static_cast<char *>(buf);
    std::size_t got = 0;
    while (got < len)
    {
        const ssize_t n = read(childOutFd_, p + got, len - got);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        got += static_cast<std::size_t>(n);
    }
    return true;
}

std::vector<double> WorkerProcessImpl::process_chunk(std::size_t taskId, const DataView &input, std::size_t &resultCols)
{
    if (input.data == nullptr && (input.rows != 0 || input.cols != 0))
        throw std::invalid_argument("Input data pointer is null for a non-empty task.");
    if (input.rows == 0 || input.cols == 0)
        throw std::invalid_argument("Task rows and cols must be > 0.");

    const std::size_t total = checked_size_product(input.rows, input.cols);
    const MessageHeader req{static_cast<uint32_t>(Protocol::Magic), kVersion,
                            static_cast<uint16_t>(MessageType::Task),
                            static_cast<uint64_t>(taskId),
                            static_cast<uint64_t>(input.rows),
                            static_cast<uint64_t>(input.cols)};
    if (!write_exact(&req, sizeof(req)))
        throw std::runtime_error("Failed to write task header to worker.");

    if (total > 0 && !write_exact(input.data, total * sizeof(double)))
        throw std::runtime_error("Failed to write task payload to worker.");

    MessageHeader resp{};
    if (!read_exact(&resp, sizeof(resp)))
        throw std::runtime_error("Worker closed output unexpectedly.");

    if (resp.magic != static_cast<uint32_t>(Protocol::Magic) || resp.version != kVersion || resp.taskId != taskId)
        throw std::runtime_error("Invalid response header from worker.");

    if (resp.type == static_cast<uint16_t>(MessageType::Error))
    {
        std::string errMsg(resp.rows, '\0');
        if (!read_exact(&errMsg[0], resp.rows))
            throw std::runtime_error("Failed to read error message from worker.");
        throw std::runtime_error("Worker error: " + errMsg);
    }

    if (resp.type != static_cast<uint16_t>(MessageType::Result))
        throw std::runtime_error("Unexpected message type from worker.");

    resultCols = resp.cols;
    if (resultCols == 0)
        throw std::runtime_error("Worker returned zero output columns.");

    const std::size_t nValues = checked_size_product(resp.rows, resultCols);
    std::vector<double> results(nValues);
    if (nValues > 0 && !read_exact(results.data(), nValues * sizeof(double)))
        throw std::runtime_error("Failed to read results from worker.");

    return results;
}

// Public API: Worker struct with callable operator
std::vector<double> Worker::operator()(std::size_t taskId, const DataView &input, std::size_t &resultCols) const
{
    static thread_local std::unique_ptr<WorkerProcessImpl> impl;
    static thread_local std::string activePythonExe;
    static thread_local std::string activeScriptPath;
    static thread_local std::size_t activeQrcLayers = 0;
    static thread_local std::size_t activeNJobs = 0;

    // Rebuild the worker process whenever its configuration changes.
    const bool configChanged =
        (activePythonExe != pythonExe) ||
        (activeScriptPath != scriptPath) ||
        (activeQrcLayers != qrcLayers) ||
        (activeNJobs != nJobs);

    if (!impl || configChanged)
    {
        impl = std::make_unique<WorkerProcessImpl>(pythonExe, scriptPath, qrcLayers, nJobs);
        activePythonExe = pythonExe;
        activeScriptPath = scriptPath;
        activeQrcLayers = qrcLayers;
        activeNJobs = nJobs;
    }

    return impl->process_chunk(taskId, input, resultCols);
}
