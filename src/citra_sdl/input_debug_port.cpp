// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "citra_sdl/input_debug_port.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <SDL.h>
#include "citra_sdl/emu_window_sdl2.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#ifdef CITRA_HAS_NATIVE_ARM
#include "core/arm/native/arm_native.h"
#endif
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/thread.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/sm/sm.h"
#include "video_core/pica/pica_core.h"
#include "input_common/keyboard.h"
#include "video_core/gpu.h"
#include "video_core/video_core.h"
#include "input_common/main.h"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

/**
 * Keyboard bindings, in Settings::NativeButton::Values order.
 *
 * Laid out for two hands on a keyboard rather than transcribed from a controller: the left hand
 * drives the circle pad on WASD (which is what a 3DS game actually walks with) with the D-pad on
 * the arrow keys beside it, and the right hand gets the face buttons on IJKL in the same diamond
 * the console has them in - I on top for X, J on the left for Y, K below for B, L on the right
 * for A. Shoulders sit above each hand's home row, Q/E for L/R and 1/3 for ZL/ZR. Start and
 * Select are Enter and Backspace, Home and Power the keys of those names, and the two buttons no
 * game reads (Debug, Gpio14) go on F1 and F2 where nothing will hit them by accident.
 *
 * The debug port's names resolve through these tables, so a remapping here carries to it.
 */
constexpr std::array<int, Settings::NativeButton::NumButtons> button_keys = {
    SDL_SCANCODE_L,         // A
    SDL_SCANCODE_K,         // B
    SDL_SCANCODE_I,         // X
    SDL_SCANCODE_J,         // Y
    SDL_SCANCODE_UP,        // Up
    SDL_SCANCODE_DOWN,      // Down
    SDL_SCANCODE_LEFT,      // Left
    SDL_SCANCODE_RIGHT,     // Right
    SDL_SCANCODE_Q,         // L
    SDL_SCANCODE_E,         // R
    SDL_SCANCODE_RETURN,    // Start
    SDL_SCANCODE_BACKSPACE, // Select
    SDL_SCANCODE_F1,        // Debug
    SDL_SCANCODE_F2,        // Gpio14
    SDL_SCANCODE_1,         // ZL
    SDL_SCANCODE_3,         // ZR
    SDL_SCANCODE_HOME,      // Home
    SDL_SCANCODE_END,       // Power
};

/// Circle pad and C-stick, as up/down/left/right/modifier. The modifier held down walks instead
/// of running (the scale below), so each stick gets the modifier under its own hand.
constexpr std::array<std::array<int, 5>, Settings::NativeAnalog::NumAnalogs> analog_keys = {{
    {SDL_SCANCODE_W, SDL_SCANCODE_S, SDL_SCANCODE_A, SDL_SCANCODE_D, SDL_SCANCODE_LSHIFT},
    {SDL_SCANCODE_KP_8, SDL_SCANCODE_KP_2, SDL_SCANCODE_KP_4, SDL_SCANCODE_KP_6,
     SDL_SCANCODE_RCTRL},
}};

/// Names the port accepts, and the scancode each presses.
int ScancodeForName(const std::string& name) {
    static const std::unordered_map<std::string, int> names = {
        {"a", button_keys[Settings::NativeButton::A]},
        {"b", button_keys[Settings::NativeButton::B]},
        {"x", button_keys[Settings::NativeButton::X]},
        {"y", button_keys[Settings::NativeButton::Y]},
        {"up", button_keys[Settings::NativeButton::Up]},
        {"down", button_keys[Settings::NativeButton::Down]},
        {"left", button_keys[Settings::NativeButton::Left]},
        {"right", button_keys[Settings::NativeButton::Right]},
        {"l", button_keys[Settings::NativeButton::L]},
        {"r", button_keys[Settings::NativeButton::R]},
        {"start", button_keys[Settings::NativeButton::Start]},
        {"select", button_keys[Settings::NativeButton::Select]},
        {"zl", button_keys[Settings::NativeButton::ZL]},
        {"zr", button_keys[Settings::NativeButton::ZR]},
        {"home", button_keys[Settings::NativeButton::Home]},
        {"power", button_keys[Settings::NativeButton::Power]},
        // The circle pad, driven through the same keys its analog binding uses.
        {"pad-up", analog_keys[Settings::NativeAnalog::CirclePad][0]},
        {"pad-down", analog_keys[Settings::NativeAnalog::CirclePad][1]},
        {"pad-left", analog_keys[Settings::NativeAnalog::CirclePad][2]},
        {"pad-right", analog_keys[Settings::NativeAnalog::CirclePad][3]},
        {"cstick-up", analog_keys[Settings::NativeAnalog::CStick][0]},
        {"cstick-down", analog_keys[Settings::NativeAnalog::CStick][1]},
        {"cstick-left", analog_keys[Settings::NativeAnalog::CStick][2]},
        {"cstick-right", analog_keys[Settings::NativeAnalog::CStick][3]},
    };
    const auto it = names.find(name);
    return it == names.end() ? -1 : it->second;
}

/// How long a `tap` holds the button. The guest samples its input once a guest frame, so a press
/// has to outlast one comfortably: measured in real time, not in polls, because the poll rate
/// depends entirely on how fast the guest happens to be running.
constexpr int DefaultTapMilliseconds = 200;

} // Anonymous namespace

void SetDefaultInputBindings() {
    auto& profile = Settings::values.current_input_profile;
    for (int i = 0; i < Settings::NativeButton::NumButtons; i++) {
        profile.buttons[i] = InputCommon::GenerateKeyboardParam(button_keys[i]);
    }
    for (int i = 0; i < Settings::NativeAnalog::NumAnalogs; i++) {
        profile.analogs[i] = InputCommon::GenerateAnalogParamFromKeys(
            analog_keys[i][0], analog_keys[i][1], analog_keys[i][2], analog_keys[i][3],
            analog_keys[i][4], 0.5f);
    }
    // The touchscreen is the window's own device, fed from the mouse (EmuWindow_SDL2's
    // HandleMouse). Without naming it here nothing creates that device and the bottom screen
    // never registers a touch, however the mouse is clicked.
    profile.touch_device = "engine:emu_window";
    LOG_INFO(Frontend,
             "Keyboard: circle pad WASD (shift walks), d-pad arrows, face buttons IJKL "
             "(X=I Y=J B=K A=L), L/R Q/E, ZL/ZR 1/3, Start Enter, Select Backspace, "
             "C-stick numpad 8456. Touch: click the bottom screen.");
}

InputDebugPort::InputDebugPort() = default;

InputDebugPort::~InputDebugPort() {
#ifndef _WIN32
    if (client_fd >= 0) {
        close(client_fd);
    }
    if (listen_fd >= 0) {
        close(listen_fd);
    }
#endif
}

#ifndef _WIN32

bool InputDebugPort::Open(u16 port) {
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        LOG_ERROR(Frontend, "Input debug port: socket failed: {}", std::strerror(errno));
        return false;
    }

    int reuse = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    fcntl(listen_fd, F_SETFL, fcntl(listen_fd, F_GETFL, 0) | O_NONBLOCK);

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    // Loopback only: this can press buttons in a game, and it asks nobody for a password.
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        listen(listen_fd, 1) != 0) {
        LOG_ERROR(Frontend, "Input debug port: could not listen on 127.0.0.1:{}: {}", port,
                  std::strerror(errno));
        close(listen_fd);
        listen_fd = -1;
        return false;
    }

    LOG_INFO(Frontend, "Input debug port listening on 127.0.0.1:{}", port);
    return true;
}

bool InputDebugPort::Poll(EmuWindow_SDL2& window) {
    if (listen_fd < 0) {
        return true;
    }

    ExpireTaps();

    if (client_fd < 0) {
        const int accepted = accept(listen_fd, nullptr, nullptr);
        if (accepted >= 0) {
            fcntl(accepted, F_SETFL, fcntl(accepted, F_GETFL, 0) | O_NONBLOCK);
            client_fd = accepted;
            pending_input.clear();
        }
    }

    if (client_fd < 0) {
        return true;
    }

    char buffer[512];
    while (true) {
        const ssize_t got = recv(client_fd, buffer, sizeof(buffer), 0);
        if (got > 0) {
            pending_input.append(buffer, static_cast<std::size_t>(got));
            continue;
        }
        if (got == 0) {
            // Client hung up.
            close(client_fd);
            client_fd = -1;
            pending_input.clear();
            return !quit_requested;
        }
        break; // EAGAIN: nothing more waiting.
    }

    std::size_t newline;
    while ((newline = pending_input.find('\n')) != std::string::npos) {
        std::string line = pending_input.substr(0, newline);
        pending_input.erase(0, newline + 1);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::string reply = Execute(line, window) + "\n";
        if (send(client_fd, reply.data(), reply.size(), MSG_NOSIGNAL) < 0) {
            close(client_fd);
            client_fd = -1;
            pending_input.clear();
            break;
        }
    }

    return !quit_requested;
}

#else

bool InputDebugPort::Open(u16) {
    LOG_ERROR(Frontend, "Input debug port is not implemented on this platform");
    return false;
}

bool InputDebugPort::Poll(EmuWindow_SDL2&) {
    return true;
}

#endif

void InputDebugPort::ExpireTaps() {
    auto* keyboard = InputCommon::GetKeyboard();
    const auto now = std::chrono::steady_clock::now();
    for (auto it = tapped_keys.begin(); it != tapped_keys.end();) {
        if (now < it->second) {
            ++it;
            continue;
        }
        if (keyboard != nullptr) {
            keyboard->ReleaseKey(it->first);
        }
        it = tapped_keys.erase(it);
    }
}

std::string InputDebugPort::Execute(const std::string& line, EmuWindow_SDL2& window) {
    std::istringstream stream(line);
    std::string command;
    stream >> command;
    std::transform(command.begin(), command.end(), command.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    auto* keyboard = InputCommon::GetKeyboard();
    if (keyboard == nullptr) {
        return "error: no keyboard device";
    }

    if (command == "press" || command == "release" || command == "tap") {
        std::string name;
        stream >> name;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const int scancode = ScancodeForName(name);
        if (scancode < 0) {
            return "error: unknown button '" + name + "'";
        }

        if (command == "release") {
            keyboard->ReleaseKey(scancode);
            tapped_keys.erase(scancode);
            return "ok";
        }

        keyboard->PressKey(scancode);
        if (command == "press") {
            tapped_keys.erase(scancode);
            return "ok";
        }

        int milliseconds = 0;
        if (!(stream >> milliseconds) || milliseconds <= 0) {
            milliseconds = DefaultTapMilliseconds;
        }
        tapped_keys[scancode] =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
        return "ok";
    }

    if (command == "touch") {
        int x = 0, y = 0;
        if (!(stream >> x >> y)) {
            return "error: touch needs x and y";
        }
        return window.TouchAt(x, y) ? "ok" : "error: not on the touch screen";
    }

    if (command == "untouch") {
        window.ReleaseTouch();
        return "ok";
    }

    if (command == "dumpsurfaces") {
        std::string dir;
        stream >> dir;
        if (dir.empty()) {
            return "error: dumpsurfaces needs a directory";
        }
        if (VideoCore::surface_dump_hook == nullptr) {
            return "error: no rasterizer registered a dump";
        }
        // On the render thread: the dump reads back through the runtime.
        auto& core = Core::System::GetInstance();
        if (!core.IsPoweredOn()) {
            return "error: no title running";
        }
        core.GPU().RunOnRenderThread([dir] {
            VideoCore::surface_dump_hook(VideoCore::surface_dump_user, dir.c_str());
        });
        return "ok";
    }

    if (command == "dumpframes") {
        std::string prefix;
        int count = 0;
        stream >> prefix >> count;
        if (prefix.empty() || count <= 0) {
            return "error: dumpframes needs a path prefix and a count";
        }
        // An optional third word "window" drops the composed canvas from the dump: footage
        // wants the presented window alone, at twice the frame rate a paired dump manages.
        std::string mode;
        stream >> mode;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        window.DumpFrames(prefix, count, mode == "window");
        return "ok dumping " + std::to_string(count);
    }

    if (command == "screenshot") {
        std::string path;
        stream >> path;
        if (path.empty()) {
            return "error: screenshot needs a path";
        }
        return window.SaveScreenshot(path) ? "ok " + path : "error: could not write " + path;
    }

    if (command == "syncsave" || command == "syncload") {
        // Savestates are refused while the GPU render thread runs; this variant stops the
        // thread (draining its queue), queues the save or load, pumps the emulation until the
        // signal executes, and restarts the thread. Lets fast frameskip-mode runs capture and
        // restore states without a synchronous boot.
        int slot = 0;
        if (!(stream >> slot) || slot < 0) {
            slot = 0;
        }
        auto& system = Core::System::GetInstance();
        if (!system.IsPoweredOn()) {
            return "error: nothing is running";
        }
        const bool was_running = system.GPU().RenderThreadRunning();
        if (was_running) {
            system.GPU().StopRenderThread();
            // The render thread released the GL context when it stopped. Serializing flushes
            // every cached surface back to guest RAM; without a current context those GL
            // downloads return uninitialized memory, which lands in guest RAM and corrupts
            // both the running game and the state being written. No-op under software.
            window.MakeCurrent();
        }
        const auto signal =
            command == "syncsave" ? Core::System::Signal::Save : Core::System::Signal::Load;
        std::string result = "ok";
        if (!system.SendSignal(signal, static_cast<u32>(slot))) {
            result = "error: another save or load is already in progress";
        } else {
            for (int i = 0; i < 400; i++) {
                if (system.RunLoop() != Core::System::ResultStatus::Success) {
                    break;
                }
            }
        }
        if (was_running) {
            window.DoneCurrent();
            system.GPU().StartRenderThread();
        }
        return result + " " + command + " slot " + std::to_string(slot);
    }

    if (command == "savestate" || command == "loadstate") {
        int slot = 0;
        if (!(stream >> slot) || slot < 0) {
            slot = 0;
        }
        return SaveOrLoadState(command == "savestate", slot);
    }

    if (command == "status") {
        return status;
    }

    if (command == "unlimited") {
        // Runtime speed toggle: "unlimited" or "unlimited off". Lets a run navigate menus at a
        // usable pace and then measure with the guest uncapped.
        std::string arg;
        stream >> arg;
        const bool on = arg != "off";
        Core::System::GetInstance().GPU().SetUnlimitedSpeed(on);
        return on ? "ok unlimited" : "ok paced";
    }

    if (command == "skip") {
        std::string arg;
        stream >> arg;
        const bool on = arg != "off";
        Core::System::GetInstance().GPU().SetFrameskipAllowed(on);
        return on ? "ok skip allowed" : "ok skip off";
    }

    if (command == "asyncflush") {
        std::string arg;
        stream >> arg;
        const bool on = arg != "off";
        Core::System::GetInstance().GPU().SetUnsafeAsyncFlush(on);
        return on ? "ok asyncflush" : "ok syncflush";
    }

#ifdef CITRA_HAS_NATIVE_ARM
    if (command == "svcring") {
        // svcring [n]: the last n guest system calls (default 64), oldest first. Read on the
        // emulation thread between guest slices, so the ring is quiescent while this formats.
        u32 want = 64;
        stream >> want;
        std::ostringstream out;
        const u32 head = Core::g_svc_ring_head.load(std::memory_order_relaxed);
        const u32 n = std::min<u32>({head, want, static_cast<u32>(Core::g_svc_ring.size())});
        for (u32 i = 0; i < n; ++i) {
            const auto& r = Core::g_svc_ring[(head - n + i) % Core::g_svc_ring.size()];
            out << "T" << r.tid << " svc" << std::hex << r.svc << " r0=" << r.r0
                << " r1=" << r.r1 << " r2=" << r.r2 << " r3=" << r.r3 << " pc=" << r.pc
                << " lr=" << r.lr << " res=" << r.result << std::dec << "\n";
        }
        return out.str();
    }
#endif

    if (command == "peek") {
        // peek <physical-addr-hex> <words-decimal>: dump guest physical memory as u32 words.
        // Runs on the emulation thread between guest slices, plain same-thread reads.
        std::string addr_s;
        u32 words = 16;
        stream >> addr_s >> words;
        const u32 addr = static_cast<u32>(std::strtoul(addr_s.c_str(), nullptr, 16));
        auto& system = Core::System::GetInstance();
        auto ref = system.Memory().GetPhysicalRef(addr);
        const u8* p = ref.GetPtr();
        if (!p) {
            return "error: bad phys addr";
        }
        words = std::min<u32>(words, 4096);
        // Clamp to the backing allocation so a bad address cannot walk the read off its end.
        words = std::min<u32>(words, static_cast<u32>(ref.GetWriteBytes(words * 4).size() / 4));
        std::ostringstream out;
        out << std::hex;
        for (u32 i = 0; i < words; ++i) {
            u32 v;
            std::memcpy(&v, p + i * 4, 4);
            out << v;
            out << (((i + 1) % 8 == 0) ? "\n" : " ");
        }
        return out.str();
    }

#ifndef _WIN32
    if (command == "cohcheck") {
        // cohcheck <phys-addr-hex> <words>: compare the emulator's view of guest memory (the
        // physical pointer) against the natively executing guest's view (the mirror mapping of
        // the corresponding linear-heap virtual address). Any differing word means the two views
        // have split - a missed or wrong mirror remap.
        std::string addr_s;
        u32 words = 256;
        stream >> addr_s >> words;
        const u32 paddr = static_cast<u32>(std::strtoul(addr_s.c_str(), nullptr, 16));
        if (paddr < 0x20000000 || paddr >= 0x30000000) {
            return "error: only FCRAM paddrs supported";
        }
        const u32 vaddr = paddr - 0x20000000 + 0x30000000;
        auto& system = Core::System::GetInstance();
        const u8* phys = system.Memory().GetPhysicalPointer(paddr);
        if (!phys) {
            return "error: bad phys addr";
        }
        const u8* mirror = reinterpret_cast<const u8*>(static_cast<uintptr_t>(vaddr));
        // Probe the mirror pages before dereferencing; an unmapped page would fault the
        // emulation thread.
        const long page = sysconf(_SC_PAGESIZE);
        const uintptr_t lo = vaddr & ~(page - 1);
        const uintptr_t hi = (vaddr + words * 4 + page - 1) & ~(page - 1);
        std::vector<unsigned char> vec((hi - lo) / page);
        if (mincore(reinterpret_cast<void*>(lo), hi - lo, vec.data()) != 0) {
            return "error: mirror range not mapped (mincore)";
        }
        words = std::min<u32>(words, 65536);
        auto ref = system.Memory().GetPhysicalRef(paddr);
        words = std::min<u32>(words, static_cast<u32>(ref.GetWriteBytes(words * 4).size() / 4));
        u32 diffs = 0;
        std::ostringstream out;
        for (u32 i = 0; i < words; ++i) {
            u32 a2, b2;
            std::memcpy(&a2, phys + i * 4, 4);
            std::memcpy(&b2, mirror + i * 4, 4);
            if (a2 != b2) {
                if (diffs < 16) {
                    out << std::hex << (paddr + i * 4) << ": phys=" << a2 << " mirror=" << b2
                        << std::dec << "\n";
                }
                ++diffs;
            }
        }
        out << "diffs=" << diffs << "/" << words;
        return out.str();
    }
#endif

    if (command == "lists") {
        // Last 48 command-list parses, oldest first: address/size, where the parse ended,
        // and whether the list wrote irq_request, raised P3D, chained, or autostopped.
        std::ostringstream out;
        const u32 head = Pica::g_list_ring_head.load(std::memory_order_relaxed);
        const u32 n = std::min<u32>({head, 48u, static_cast<u32>(Pica::g_list_ring.size())});
        for (u32 i = 0; i < n; ++i) {
            const auto& r = Pica::g_list_ring[(head - n + i) % Pica::g_list_ring.size()];
            out << std::hex << r.addr << "+" << r.size << std::dec << " end=" << r.end_index
                << "/" << r.end_length << " irqw=" << r.irqw << " p3d=" << r.p3d
                << " chain=" << r.chains << " stop=" << r.stopped << " lc=" << std::hex
                << r.last_chain_addr << "+" << r.last_chain_size << std::dec << "\n";
        }
        return out.str();
    }

    if (command == "gsp") {
        // Snapshot of the guest/GSP handshake state, for reading at a wedge. The port is polled
        // on the emulation thread, the same thread every GSP shared-page access runs on, so the
        // reads are ordinary same-thread reads.
        auto& system = Core::System::GetInstance();
        auto gsp = system.ServiceManager().GetService<Service::GSP::GSP_GPU>("gsp::Gpu");
        if (!gsp) {
            return "error: gsp service not up";
        }
        std::ostringstream out;
        const u32 rights = gsp->GetActiveThreadId();
        out << "rights=" << rights;
        if (rights != std::numeric_limits<u32>::max()) {
            if (auto* cb = gsp->GetCommandBuffer(rights)) {
                out << " cmd[idx=" << cb->index.Value() << " n=" << cb->number_commands.Value()
                    << " status=" << cb->status.Value() << " stop=" << cb->should_stop.Value()
                    << " ids=";
                for (u32 i = 0; i < cb->number_commands && i < 15; ++i) {
                    out << static_cast<u32>(cb->commands[(cb->index + i) % 0xF].id.Value()) << ",";
                }
                out << "]";
            }
            if (auto* irq = gsp->GetInterruptRelayQueue(rights)) {
                out << " irq[idx=" << static_cast<int>(irq->index)
                    << " n=" << static_cast<int>(irq->number_interrupts)
                    << " err=" << static_cast<int>(irq->error_code)
                    << " cfg=" << static_cast<int>(irq->config)
                    << " missedPDC=" << irq->missed_PDC0 << "/" << irq->missed_PDC1 << " slots=";
                for (int i = 0; i < irq->number_interrupts && i < 16; ++i) {
                    out << static_cast<int>(irq->slot[(irq->index + i) % 0x34]) << ",";
                }
                out << "]";
            }
        }
        out << " pica[cl=" << Pica::g_pica_probe[0].load()
            << " irqw=" << Pica::g_pica_probe[1].load()
            << " p3d=" << Pica::g_pica_probe[2].load()
            << " nomatch=" << Pica::g_pica_probe[3].load()
            << " chain=" << Pica::g_pica_probe[4].load()
            << " stop=" << Pica::g_pica_probe[5].load() << "]";
        out << " probe[pdc=" << Service::GSP::g_gsp_probe[0].load()
            << " ign=" << Service::GSP::g_gsp_probe[1].load()
            << " pq=" << Service::GSP::g_gsp_probe[2].load()
            << " miss=" << Service::GSP::g_gsp_probe[3].load()
            << " oq=" << Service::GSP::g_gsp_probe[4].load()
            << " nosess=" << Service::GSP::g_gsp_probe[5].load()
            << " noevt=" << Service::GSP::g_gsp_probe[6].load()
            << " full=" << Service::GSP::g_gsp_probe[7].load() << "]";
        for (u32 core = 0; core < system.GetNumCores(); ++core) {
            for (const auto& t : system.Kernel().GetThreadManager(core).GetThreadList()) {
                out << "\nT" << t->thread_id << " core" << core << " prio" << t->current_priority
                    << " st" << static_cast<int>(t->status) << " pc=" << std::hex
                    << t->context.GetProgramCounter() << " lr=" << t->context.GetLinkRegister()
                    << std::dec;
                if (!t->wait_objects.empty()) {
                    out << " wait=";
                    for (const auto& o : t->wait_objects) {
                        out << o->GetName() << ",";
                    }
                }
                if (t->status == Kernel::ThreadStatus::WaitArb) {
                    out << " arb@" << std::hex << t->wait_address << std::dec;
                }
            }
        }
        return out.str();
    }

    if (command == "quit") {
        quit_requested = true;
        return "ok";
    }

    return "error: unknown command '" + command +
           "'; try press/release/tap/touch/untouch/savestate/loadstate/screenshot/status/"
           "unlimited/skip/asyncflush/gsp/lists/svcring/peek/cohcheck/quit";
}

std::string InputDebugPort::SaveOrLoadState(bool save, int slot) {
    auto& system = Core::System::GetInstance();
    if (!system.IsPoweredOn()) {
        return "error: nothing is running";
    }

    // Savestates are taken by the emulator itself at a quiescent point rather than from here, so
    // this only asks; the answer arrives in the log.
    const auto signal = save ? Core::System::Signal::Save : Core::System::Signal::Load;
    if (!system.SendSignal(signal, static_cast<u32>(slot))) {
        return "error: another save or load is already in progress";
    }
    return (save ? "ok saving slot " : "ok loading slot ") + std::to_string(slot);
}
