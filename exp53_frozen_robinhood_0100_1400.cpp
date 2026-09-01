/* EXP53 100-1400 Robin Hood integration experiment.

   This is deliberately a wrapper around the frozen kernel, not a replacement
   for its arithmetic.  The hash table only detects exact-bit duplicate inputs
   inside the batch.  Each unique value is evaluated once by the frozen kernel
   and scattered back to duplicate positions.

   Correctness invariant:
   - The original full-32 prefix is deduplicated.
   - The compacted unique list is padded to a multiple of 32 before calling the
     frozen kernel, so every deduplicated prefix value is evaluated by the same
     AVX-512 arithmetic used by the original frozen prefix.
   - The original <32 tail is passed through the frozen kernel unchanged, thus
     preserving its scalar-exp tail semantics exactly.

   No heap allocation is performed.  The table has 2048 slots; at n<=1400 the
   largest hashed prefix is 1376 entries (<=67.2% load).
*/
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

namespace {
constexpr size_t RH_CAP = 2048;
constexpr size_t RH_MASK = RH_CAP - 1;
constexpr size_t MAX_N = 1400;

struct RHWorkspace {
    uint64_t keys[RH_CAP];
    uint16_t ids[RH_CAP];
    uint16_t dist[RH_CAP];
    uint16_t tag[RH_CAP];
    uint16_t epoch = 0;
    uint16_t unique_count = 0;
    uint16_t map[MAX_N];
    double unique_in[MAX_N + 32];
    double unique_out[MAX_N + 32];
};

thread_local RHWorkspace ws;

static inline uint64_t bits_of(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

/* One-multiply 64-bit mixer.  Exact key comparison still decides equality. */
static inline uint64_t rh_hash(uint64_t x) {
    x ^= x >> 32;
    x *= UINT64_C(0xd6e8feb86659fd93);
    x ^= x >> 32;
    return x;
}

static inline void begin_table() {
    ++ws.epoch;
    if (ws.epoch == 0) {
        std::memset(ws.tag, 0, sizeof(ws.tag));
        ws.epoch = 1;
    }
    ws.unique_count = 0;
}

/* Return stable unique-id for key.  The new id is reserved lazily: a duplicate
   normally exits without touching the compacted-value arrays. */
static inline uint16_t rh_find_or_insert(uint64_t key, double value) {
    const uint16_t new_id = ws.unique_count;
    uint64_t cur_key = key;
    uint16_t cur_id = new_id;
    uint16_t cur_dist = 0;
    size_t pos = static_cast<size_t>(rh_hash(key)) & RH_MASK;
    bool new_reserved = false;

    for (;;) {
        if (ws.tag[pos] != ws.epoch) {
            if (!new_reserved) {
                ws.unique_in[new_id] = value;
                ++ws.unique_count;
                new_reserved = true;
            }
            ws.tag[pos] = ws.epoch;
            ws.keys[pos] = cur_key;
            ws.ids[pos] = cur_id;
            ws.dist[pos] = cur_dist;
            return new_id;
        }

        if (ws.keys[pos] == cur_key) {
            /* Before the first Robin-Hood swap this is the normal duplicate
               hit.  After a swap an equal displaced key cannot exist in a
               valid table because every unique key has exactly one slot. */
            return ws.ids[pos];
        }

        if (ws.dist[pos] < cur_dist) {
            if (!new_reserved) {
                ws.unique_in[new_id] = value;
                ++ws.unique_count;
                new_reserved = true;
            }
            std::swap(cur_key, ws.keys[pos]);
            std::swap(cur_id, ws.ids[pos]);
            std::swap(cur_dist, ws.dist[pos]);
        }

        pos = (pos + 1) & RH_MASK;
        ++cur_dist;
    }
}
} // namespace

extern "C" __attribute__((noinline))
void exp53_frozen_robinhood(double *out, const double *in, size_t n) {
    if (n == 0) return;
    if (n > MAX_N) {
        /* Experiment is intentionally scoped to 100..1400. */
        exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        return;
    }

    const size_t prefix = n & ~size_t(31);
    if (prefix == 0) {
        exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        return;
    }

    begin_table();
    for (size_t i = 0; i < prefix; ++i)
        ws.map[i] = rh_find_or_insert(bits_of(in[i]), in[i]);

    const size_t u = ws.unique_count;
    const size_t padded = (u + 31) & ~size_t(31);
    for (size_t i = u; i < padded; ++i)
        ws.unique_in[i] = ws.unique_in[0];

    exp53_n2_vmstyle_u4_0381_frozen(ws.unique_out, ws.unique_in, padded);
    for (size_t i = 0; i < prefix; ++i)
        out[i] = ws.unique_out[ws.map[i]];

    if (prefix < n)
        exp53_n2_vmstyle_u4_0381_frozen(out + prefix, in + prefix, n - prefix);
}
