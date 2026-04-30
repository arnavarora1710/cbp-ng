#define USE_META
#define RESET_UBITS

#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;

struct predictor_base {
    predictor* parent = nullptr;
    virtual void bind_parent(predictor* parent) = 0;
    virtual val<1> predict(val<64> inst_pc) = 0;
    virtual void update_condbr(val<64> branch_pc, val<1> taken, val<64> next_pc) = 0;
};

template <u64 PHT_LOG = 6, u64 HISTORY_LEN = 4>
struct two_level : predictor_base {
    ram<val<HISTORY_LEN>, (1ull << PHT_LOG)> pattern_table;
    ram<val<2>, (1ull << HISTORY_LEN)> branch_hist_table;
    reg<2> counter;

    void bind_parent(predictor* parent) override {
        this->parent = parent;
    }

    val<1> predict(val<64> inst_pc) override {
        val<PHT_LOG> index = (inst_pc >> 6).make_array(val<PHT_LOG>{}).fold_xor();
        val<HISTORY_LEN> pattern = pattern_table.read(index);
        parent->need_extra_cycle(1);
        counter = branch_hist_table.read(pattern);
        return counter >> 1;
    }

    inline val<2> update_counter(val<2> counter, val<1> taken) {
        val<2> increased = select(counter == 3, counter, val<2>{counter + 1});
        val<2> decreased = select(counter == 0, counter, val<2>{counter - 1});
        return select(taken, increased, decreased);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken,
                       [[maybe_unused]] val<64> next_pc) override {
        val<2> new_counter = update_counter(counter, taken);
        val<1> is_update_needed = val<1>{new_counter != counter};

        val<PHT_LOG> index = (branch_pc >> 6).make_array(val<PHT_LOG>{}).fold_xor();

        parent->need_extra_cycle(1);
        val<HISTORY_LEN> pattern = pattern_table.read(index);

        parent->need_extra_cycle(is_update_needed);

        execute_if(is_update_needed, [&] {
            branch_hist_table.write(pattern, new_counter);
        });

        parent->need_extra_cycle(1);
        pattern_table.write(index, (pattern << 1) | taken);
    }
};

template <u64 PHT_LOG = 10, u64 HISTORY_LEN = 16>
struct local_two_level {
    ram<val<HISTORY_LEN>, (1ull << PHT_LOG)> pattern_table;
    ram<val<2>, (1ull << HISTORY_LEN)> branch_hist_table;
    reg<2> counter;

    predictor* parent = nullptr;

    void bind_parent(predictor *p) {
        parent = p;
    }

    val<1> predict(val<64> inst_pc) {
        val<PHT_LOG> index = (inst_pc >> 6).make_array(val<PHT_LOG>{}).fold_xor();
        val<HISTORY_LEN> pattern = pattern_table.read(index);
        parent->need_extra_cycle(1);
        counter = branch_hist_table.read(pattern);
        return counter >> 1;
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

        parent->need_extra_cycle(1);
        val<HISTORY_LEN> pattern = pattern_table.read(index);

        parent->need_extra_cycle(is_update_needed);

        execute_if(is_update_needed, [&] {
            branch_hist_table.write(pattern, new_counter);
        });

        parent->need_extra_cycle(1);
        pattern_table.write(index, (pattern << 1) | taken);
    }
};

template <u64 HISTORY_LEN = 16>
struct global_pred {
    // global shift register
    reg<HISTORY_LEN> gshr;
    // pattern history table
    ram<val<2>, (1ull << HISTORY_LEN)> pattern_table;

    predictor* parent = nullptr;

    void bind_parent(predictor* p) {
        parent = p;
    }

    val<1> predict ([[maybe_unused]] val<64> inst_pc) {
        val<2> counter = pattern_table.read(gshr);
        parent->need_extra_cycle(1);
        return counter >> 1;
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
        parent->need_extra_cycle(is_update_needed);
        execute_if(is_update_needed, [&] {
            pattern_table.write(gshr, new_counter);
        });
        gshr = (gshr << 1) | taken;
    }
};

template <u64 PHT_LOG = 10, u64 LOCAL_HISTORY_LEN = 16, u64 GLOBAL_HISTORY_LEN = 16>
struct tournament : predictor_base {
    local_two_level<PHT_LOG, LOCAL_HISTORY_LEN> local_predictor;
    global_pred<GLOBAL_HISTORY_LEN> global_predictor;

    reg<2> chooser; // 2 bit saturating counter

    reg<1> last_local;
    reg<1> last_global;

    void bind_parent(predictor* parent) override {
        this->parent = parent;
    }

    val<1> predict(val<64> inst_pc) override {
        val<1> local_pred = local_predictor.predict(inst_pc);
        val<1> global_pred = global_predictor.predict(inst_pc);
        last_local = local_pred;
        parent->need_extra_cycle(1);
        last_global = global_pred;

        val<1> use_local = chooser >> 1;
        return select(use_local, local_pred, global_pred);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken,
                       [[maybe_unused]] val<64> next_pc) override {
        local_predictor.update_condbr(branch_pc, taken, next_pc);
        global_predictor.update_condbr(branch_pc, taken, next_pc);

        val<1> local_correct = (last_local == taken);
        val<1> global_correct = (last_global == taken);

        val<1> favor_local = local_correct & ~global_correct;
        val<1> favor_global = ~local_correct & global_correct;

        execute_if(favor_local, [&] {
            chooser = select(chooser == 3, chooser, val<2>{chooser + 1});
            parent->need_extra_cycle(1);
        });

        execute_if(favor_global, [&] {
            chooser = select(chooser == 0, chooser, val<2>{chooser - 1});
            parent->need_extra_cycle(1);
        });
    }
};
