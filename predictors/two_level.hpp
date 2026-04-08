#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

template <class = void>
struct two_level : predictor {
    ram<val<4>, 64> pattern_table;
    ram<val<2>, 16> branch_hist_table;
    reg<2> counter;

    val<1> predict(val<64> inst_pc) {
        val<6> index = (inst_pc >> 6).make_array(val<6>{}).fold_xor();
        val<4> pattern = pattern_table.read(index);
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

        val<6> index = (branch_pc >> 6).make_array(val<6>{}).fold_xor();

        need_extra_cycle(1);
        val<4> pattern = pattern_table.read(index);

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
