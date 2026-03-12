#include "moonshine-cpp.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

static const int PORT = 9900;
static std::atomic<bool> g_running{true};

static void signalHandler(int) { g_running = false; }

// ============================================================
// Thread-safe audio chunk queue (fix #2: decouple recv from inference)
// ============================================================
class AudioQueue {
public:
    void push(std::vector<float>&& chunk) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(chunk));
        m_cv.notify_one();
    }

    bool pop(std::vector<float>& out, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [this] { return !m_queue.empty(); }))
            return false;
        out = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::queue<std::vector<float>>().swap(m_queue);
    }

private:
    std::queue<std::vector<float>> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

// ============================================================
// Event listener — send JSON to client
// ============================================================
class JsonSender : public moonshine::TranscriptEventListener {
public:
    void setFd(int fd) { m_fd.store(fd, std::memory_order_release); }

    void onLineTextChanged(const moonshine::LineTextChanged& event) override {
        sendJson(event.line.text, true);
    }

    void onLineCompleted(const moonshine::LineCompleted& event) override {
        sendJson(event.line.text, false);
    }

private:
    void sendJson(const std::string& text, bool partial) {
        int fd = m_fd.load(std::memory_order_acquire);
        if (fd < 0) return;

        std::string escaped;
        escaped.reserve(text.size() + 16);
        for (char c : text) {
            if (c == '"') escaped += "\\\"";
            else if (c == '\\') escaped += "\\\\";
            else if (c == '\n') escaped += "\\n";
            else escaped += c;
        }

        char json[4096];
        int len = snprintf(json, sizeof(json),
                 "{\"text\":\"%s\",\"%s\":true}\n",
                 escaped.c_str(),
                 partial ? "partial" : "final");
        if (len >= (int)sizeof(json)) len = sizeof(json) - 1;

        ::send(fd, json, len, MSG_NOSIGNAL);
    }

    std::atomic<int> m_fd{-1};
};

// ============================================================
// Main — single-client TCP server
// ============================================================
int main(int argc, char** argv) {
    const char* modelPath = (argc > 1) ? argv[1] :
        "/home/tqt/.cache/moonshine_voice/download.moonshine.ai/model/tiny-streaming-en/quantized";
    int port = (argc > 2) ? atoi(argv[2]) : PORT;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGPIPE, SIG_IGN);

    // Load model once at startup
    printf("[SERVER] Loading model: %s\n", modelPath);
    moonshine::Transcriber transcriber(modelPath,
        moonshine::ModelArch::TINY_STREAMING, 0.1);

    JsonSender sender;
    transcriber.addListener(&sender);

    printf("[SERVER] Listening on port %d\n", port);

    int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listenFd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); ::close(listenFd); return 1;
    }
    listen(listenFd, 1);

    printf("[SERVER] Ready. Waiting for client...\n");

    AudioQueue audioQueue;

    while (g_running) {
        // Poll with timeout so we can check g_running periodically
        pollfd pfd = { listenFd, POLLIN, 0 };
        int pr = poll(&pfd, 1, 500);  // 500ms timeout
        if (pr <= 0) continue;        // timeout or error — recheck g_running

        sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        int clientFd = accept(listenFd, (sockaddr*)&clientAddr, &clientLen);
        if (clientFd < 0) {
            if (g_running) perror("accept");
            continue;
        }

        printf("[SERVER] Client connected (fd=%d)\n", clientFd);
        int flag = 1;
        setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        setsockopt(clientFd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
        sender.setFd(clientFd);

        transcriber.start();
        audioQueue.clear();

        // Recv thread: read from socket, push into queue
        std::atomic<bool> clientDone{false};
        std::thread recvThread([&] {
            constexpr int BUF_SAMPLES = 1600;  // 100ms @ 16kHz
            float buf[BUF_SAMPLES];
            while (!clientDone && g_running) {
                int bytesNeeded = BUF_SAMPLES * sizeof(float);
                int bytesRead = 0;
                while (bytesRead < bytesNeeded && !clientDone && g_running) {
                    int n = recv(clientFd, (char*)buf + bytesRead,
                                 bytesNeeded - bytesRead, 0);
                    if (n <= 0) { clientDone = true; break; }
                    bytesRead += n;
                }
                if (clientDone) break;
                audioQueue.push(std::vector<float>(buf, buf + BUF_SAMPLES));
            }
        });

        // Main thread: pop from queue and feed transcriber
        std::vector<float> samples;
        while (g_running) {
            if (audioQueue.pop(samples, 100)) {
                transcriber.addAudio(samples, 16000);
            } else if (clientDone) {
                break;  // queue empty and client disconnected
            }
        }

        recvThread.join();
        transcriber.stop();
        sender.setFd(-1);
        ::close(clientFd);
        printf("[SERVER] Client disconnected\n");
        printf("[SERVER] Waiting for next client...\n");
    }

    ::close(listenFd);
    printf("[SERVER] Shutdown\n");
    return 0;
}
