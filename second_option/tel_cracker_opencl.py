import time
import numpy as np
import pyopencl as cl


KERNEL_SRC = r"""
inline uint rol(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

inline void sha1_len11(const uchar msg11[11], __private uint out[5]) {
    uint h0 = 0x67452301U;
    uint h1 = 0xEFCDAB89U;
    uint h2 = 0x98BADCFEU;
    uint h3 = 0x10325476U;
    uint h4 = 0xC3D2E1F0U;

    uchar m[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) m[i] = (uchar)0;

    #pragma unroll
    for (int i = 0; i < 11; i++) m[i] = msg11[i];

    m[11] = (uchar)0x80;
    m[63] = (uchar)0x58; // 11*8 bits

    uint W[80];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int j = i * 4;
        W[i] = ((uint)m[j] << 24) | ((uint)m[j+1] << 16) | ((uint)m[j+2] << 8) | ((uint)m[j+3]);
    }
    for (int t = 16; t < 80; t++) {
        W[t] = rol(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1);
    }

    uint a = h0, b = h1, c = h2, d = h3, e = h4;

    for (int t = 0; t < 80; t++) {
        uint f, k;
        if (t < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999U; }
        else if (t < 40) { f = b ^ c ^ d;            k = 0x6ED9EBA1U; }
        else if (t < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCU; }
        else             { f = b ^ c ^ d;            k = 0xCA62C1D6U; }

        uint temp = rol(a, 5) + f + e + k + W[t];
        e = d;
        d = c;
        c = rol(b, 30);
        b = a;
        a = temp;
    }

    out[0] = h0 + a;
    out[1] = h1 + b;
    out[2] = h2 + c;
    out[3] = h3 + d;
    out[4] = h4 + e;
}

__kernel void crack_sha1_mask_8_10digits(
    const ulong start_idx,
    const ulong count,
    __constant uint *target5,
    __global volatile int *found_flag,
    __global ulong *found_idx
) {
    ulong gid = (ulong)get_global_id(0);
    if (gid >= count) return;
    if (*found_flag) return;

    ulong idx = start_idx + gid;
    if (idx > 9999999999UL) return;

    uchar msg[11];
    msg[0] = (uchar)'8';

    ulong x = idx;
    #pragma unroll
    for (int pos = 10; pos >= 1; pos--) {
        uint digit = (uint)(x % 10UL);
        msg[pos] = (uchar)('0' + digit);
        x /= 10UL;
    }

    uint digest[5];
    sha1_len11(msg, digest);

    int ok = 1;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (digest[i] != target5[i]) { ok = 0; break; }
    }

    if (ok) {
        int old = atomic_cmpxchg((volatile __global int *)found_flag, 0, 1);
        if (old == 0) {
            *found_idx = idx;
        }
    }
}
"""


def _sha1_hex_to_u32be5(sha1_hex: str) -> np.ndarray:
    s = sha1_hex.strip().lower()
    if len(s) != 40 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"Expected 40-hex SHA1 digest, got: {sha1_hex!r}")
    raw = bytes.fromhex(s)
    return np.frombuffer(raw, dtype=">u4").astype(np.uint32)


def _pick_device(prefer_cpu: bool = True) -> cl.Device:
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        devices.extend(p.get_devices())
    if not devices:
        raise RuntimeError("No OpenCL devices found")

    if prefer_cpu:
        cpu = [d for d in devices if d.type & cl.device_type.CPU]
        if cpu:
            return cpu[0]
    return devices[0]


def crack_one_hash(
    target_sha1_hex: str,
    start_idx: int,
    end_idx_exclusive: int,
    launch_size: int = 2_097_152,
    chunk_size: int = 200_000_000,
    progress_every: int = 0,
    prefer_cpu: bool = True
):
    dev = _pick_device(prefer_cpu=prefer_cpu)
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL_SRC).build()

    target5 = _sha1_hex_to_u32be5(target_sha1_hex)

    mf = cl.mem_flags
    target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target5)

    found_flag_host = np.array([0], dtype=np.int32)
    found_idx_host = np.array([0], dtype=np.uint64)

    found_flag_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_flag_host)
    found_idx_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_idx_host)

    def idx_to_phone(i: int) -> str:
        return "8" + f"{i:010d}"

    # reset
    found_flag_host[0] = 0
    found_idx_host[0] = 0
    cl.enqueue_copy(queue, found_flag_buf, found_flag_host)
    cl.enqueue_copy(queue, found_idx_buf, found_idx_host)
    queue.finish()

    total_start = time.time()
    tested = 0
    launch_counter = 0

    idx = int(start_idx)
    end = int(end_idx_exclusive)

    while idx < end:
        chunk_end = min(idx + chunk_size, end)
        local_start = idx

        while local_start < chunk_end:
            remaining = chunk_end - local_start
            this_launch = min(remaining, launch_size)

            prg.crack_sha1_mask_8_10digits(
                queue,
                (int(this_launch),),
                None,
                np.uint64(local_start),
                np.uint64(this_launch),
                target_buf,
                found_flag_buf,
                found_idx_buf
            )

            tested += int(this_launch)
            local_start += int(this_launch)
            launch_counter += 1

            # останавливаемся сразу, если нашли
            cl.enqueue_copy(queue, found_flag_host, found_flag_buf)
            queue.finish()
            if found_flag_host[0] != 0:
                cl.enqueue_copy(queue, found_idx_host, found_idx_buf)
                queue.finish()
                found_idx = int(found_idx_host[0])
                return idx_to_phone(found_idx), found_idx

            if progress_every and (launch_counter % progress_every == 0):
                elapsed = time.time() - total_start
                mh_s = (tested / elapsed) / 1e6 if elapsed > 0 else 0.0
                print(f"Launches={launch_counter:,} Tested={tested:,} Elapsed={elapsed:.1f}s Speed≈{mh_s:.1f} MH/s idx={local_start:,}")

        idx = chunk_end

    return None, None


def crack_hashes(
    sha1_list: list[str],
    phone_prefix_after_8: str = "9", # 89xxxxxxxxx
    prefer_cpu: bool = True
) -> dict[str, str]:
    """
    Возвращает словарь: sha1_hex -> phone (11 цифр, начинается с 8).
    По умолчанию ищем только по маске 89xxxxxxxxx, т.к. по датасету видно, что все телефоны начинаются на 89.
    """
    if not phone_prefix_after_8.isdigit():
        raise ValueError("phone_prefix_after_8 must be digits")
    k = len(phone_prefix_after_8)
    if k > 10:
        raise ValueError("prefix too long")

    start_idx = int(phone_prefix_after_8) * (10 ** (10 - k))
    end_idx = (int(phone_prefix_after_8) + 1) * (10 ** (10 - k))

    result: dict[str, str] = {}
    for h in sha1_list:
        hh = str(h).strip().lower()
        phone, _ = crack_one_hash(
            target_sha1_hex=hh,
            start_idx=start_idx,
            end_idx_exclusive=end_idx,
            prefer_cpu=prefer_cpu
        )
        if phone:
            result[hh] = phone
        else:
            result[hh] = ""
    return result
