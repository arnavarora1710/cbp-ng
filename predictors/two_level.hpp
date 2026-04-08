#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

//   - PHT: indexed by hashed PC, stores HISTORY_LEN bits of local history
//   - BHT: indexed directly by that local history, stores 2-bit counters

// Knobs:
//   - PHT_LOG: log2(PHT entries)
//   - HISTORY_LEN: local history length (also implies BHT has 2^HISTORY_LEN entries)
template <u64 PHT_LOG = 6, u64 HISTORY_LEN = 4>
struct two_level : predictor {
    ram<val<HISTORY_LEN>, (1ull << PHT_LOG)> pattern_table;
    ram<val<2>, (1ull << HISTORY_LEN)> branch_hist_table;
    reg<2> counter;

    val<1> predict(val<64> inst_pc) {
        val<PHT_LOG> index = (inst_pc >> 6).make_array(val<PHT_LOG>{}).fold_xor();
        val<HISTORY_LEN> pattern = pattern_table.read(index);
        need_extra_cycle(1);
        counter = branch_hist_table.read(pattern);
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
        val<2> new_counter = update_counter(counter, taken);
        val<1> is_update_needed = val<1>{new_counter != counter};

        val<PHT_LOG> index = (branch_pc >> 6).make_array(val<PHT_LOG>{}).fold_xor();

        need_extra_cycle(1);
        val<HISTORY_LEN> pattern = pattern_table.read(index);

        need_extra_cycle(is_update_needed);

        execute_if(is_update_needed, [&] {
            branch_hist_table.write(pattern, new_counter);
        });

        need_extra_cycle(1);
        pattern_table.write(index, (pattern << 1) | taken);
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
