#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

template <u64 HISTORY_LEN = 16>
struct global_pred : predictor {
    // global shift register
    reg<HISTORY_LEN> gshr;
    // pattern history table
    ram<val<2>, (1ull << HISTORY_LEN)> pattern_table;

    val<1> predict([[maybe_unused]] val<64> inst_pc) {
        val<2> counter = pattern_table.read(gshr);
        need_extra_cycle(1);
        return counter >> 1;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc) {
        return hard<0>{};
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc) {
        return predict(inst_pc);
    }

    inline val<2> update_counter(val<2> counter, val<1> taken) {
        val<2> increased = select(counter == 3, counter, val<2>{counter + 1});
        val<2> decreased = select(counter == 0, counter, val<2>{counter - 1});
        return select(taken, increased, decreased);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken,
                       [[maybe_unused]] val<64> next_pc) {
        val<2> old_counter = pattern_table.read(gshr);
        val<2> new_counter = update_counter(old_counter, taken);
        val<1> is_update_needed = val<1>{new_counter != old_counter};
        need_extra_cycle(is_update_needed);
        execute_if(is_update_needed, [&] {
            pattern_table.write(gshr, new_counter);
        });
        gshr = (gshr << 1) | taken;
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) {
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc) {
        return hard<0>{};
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc) {
        return hard<0>{};
    }
};
