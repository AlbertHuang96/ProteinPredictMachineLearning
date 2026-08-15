#include <cstdint>

#include "Tensor.h"

//typedef uint32_t bitset_t;
using bitset_t = uint64_t;

namespace ppml {

#define BITSET_SHR 5 
// log2(sizeof(bitset_t)*8)
#define BITSET_MASK (sizeof(bitset_t)*8 - 1)
static size_t bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}
static inline bool bitset_get(const bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}
static inline void bitset_set(bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}
static inline void bitset_clear(bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}



#define HASHSET_FULL ((size_t)-1)
#define HASHSET_ALREADY_EXISTS ((size_t)-2)
struct HashSet {
    size_t     size;
    bitset_t * used;       // whether or not the keys are in use i.e. set
    void    ** keys;
    //struct Tensor ** keys; // actual tensors in the set, keys[i] is only defined if bitset_get(used, i)
    //error: template argument required for ‘struct Tensor
};


// hash function for ggml_tensor
//void* p
//struct Tensor * p
static inline size_t hash(void * p) {
    // the last 4 bits are always zero due to alignment
    //return (size_t)(uintptr_t)p >> 4;
    return (size_t)p ^ ((size_t)p >> 32);
}
 
//void *
//struct Tensor *
static size_t hash_find(const struct HashSet * hash_set, void * key) {
    size_t h = hash(key) % hash_set->size;
    // linear probing
    size_t i = h;
    while (bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {
            // visited all hash table entries -> not found
            return HASHSET_FULL;
        }
    }
    return i;
}

static inline void hash_set_reset(struct HashSet * hash_set) {  
    memset(hash_set->used, 0, sizeof(bitset_t) * bitset_size(hash_set->size));  
}


static bool hash_contains(const struct HashSet * hash_set, void * key) {
    size_t i = hash_find(hash_set, key);
    return i != HASHSET_FULL && bitset_get(hash_set->used, i);
}
static size_t hash_insert(struct HashSet * hash_set, void * key) {
    size_t h = hash(key) % hash_set->size;
    // linear probing
    size_t i = h;
    do {
        if (!bitset_get(hash_set->used, i)) {
            bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            //error: invalid conversion from ‘const void*’ to ‘void*’
            return i;
        }
        if (hash_set->keys[i] == key) {
            return HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);
    // visited all hash table entries -> not found
    // GGML_ABORT("fatal error");
}

} // namespace ppml