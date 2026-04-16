#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

template <class = void>
struct tournament : predictor {
    static constexpr u64 GHR_LEN = 8;
    static constexpr u64 GLOBAL_TABLE_SIZE = 1ull << GHR_LEN;

    static constexpr u64 LOCAL_PC_BITS = 8;
    static constexpr u64 LOCAL_HISTORY_LEN = 6;
    static constexpr u64 LOCAL_HISTORY_TABLE_SIZE = 1ull << LOCAL_PC_BITS;
    static constexpr u64 LOCAL_PHT_SIZE = 1ull << LOCAL_HISTORY_LEN;

    static constexpr u64 CHOOSER_PC_BITS = 8;
    static constexpr u64 CHOOSER_TABLE_SIZE = 1ull << CHOOSER_PC_BITS;

    reg<GHR_LEN> ghr = 0;

    // Global predictor table
    ram<val<2>, GLOBAL_TABLE_SIZE> global_pht;

    // Local predictor tables
    ram<val<LOCAL_HISTORY_LEN>, LOCAL_HISTORY_TABLE_SIZE> local_history_table;
    ram<val<2>, LOCAL_PHT_SIZE> local_pht;

    // Chooser table
    ram<val<2>, CHOOSER_TABLE_SIZE> chooser;

    inline val<2> update_counter(val<2> counter, val<1> taken) {
        val<2> increased = select(counter == val<2>{3}, val<2>{3}, val<2>{counter + val<2>{1}});
        val<2> decreased = select(counter == val<2>{0}, val<2>{0}, val<2>{counter - val<2>{1}});
        return select(taken, increased, decreased);
    }

    inline val<1> counter_prediction(val<2> counter) {
        return counter >> hard<1>{};
    }

    inline val<GHR_LEN> get_global_index(val<64> pc) {
        val<GHR_LEN> pc_bits = (pc >> hard<2>{}) & val<GHR_LEN>{(1ull << GHR_LEN) - 1};
        return ghr ^ pc_bits;
    }

    inline val<LOCAL_PC_BITS> get_local_pc_index(val<64> pc) {
        return (pc >> hard<2>{}) & val<LOCAL_PC_BITS>{(1ull << LOCAL_PC_BITS) - 1};
    }

    inline val<CHOOSER_PC_BITS> get_chooser_index(val<64> pc) {
        return (pc >> hard<2>{}) & val<CHOOSER_PC_BITS>{(1ull << CHOOSER_PC_BITS) - 1};
    }

    inline void stall_stage() {
        need_extra_cycle(1);
    }

    val<1> predict1(val<64> inst_pc) override {
        val<GHR_LEN> global_idx = get_global_index(inst_pc);
        val<2> global_ctr = global_pht.read(global_idx);

        // Disable reuse for now until a real reuse path is implemented
        reuse_prediction(val<1>{0});

        return counter_prediction(global_ctr);
    }

    val<1> reuse_predict1(val<64> inst_pc) override {
        return predict1(inst_pc);
    }

    val<1> predict2(val<64> inst_pc) override {
        // Local prediction
        val<LOCAL_PC_BITS> local_pc_idx = get_local_pc_index(inst_pc);
        val<LOCAL_HISTORY_LEN> local_hist = local_history_table.read(local_pc_idx);

        stall_stage();
        val<2> local_ctr = local_pht.read(local_hist);
        val<1> local_pred = counter_prediction(local_ctr);

        // Global prediction
        stall_stage();
        val<GHR_LEN> global_idx = get_global_index(inst_pc);
        val<2> global_ctr = global_pht.read(global_idx);
        val<1> global_pred = counter_prediction(global_ctr);

        // Chooser
        stall_stage();
        val<CHOOSER_PC_BITS> chooser_idx = get_chooser_index(inst_pc);
        val<2> chooser_ctr = chooser.read(chooser_idx);
        val<1> use_global = counter_prediction(chooser_ctr);

        reuse_prediction(val<1>{0});

        return select(use_global, global_pred, local_pred);
    }

    val<1> reuse_predict2(val<64> inst_pc) override {
        return predict2(inst_pc);
    }

    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc) override {
        // ----- Local predictor -----
        val<LOCAL_PC_BITS> local_pc_idx = get_local_pc_index(branch_pc);
        val<LOCAL_HISTORY_LEN> old_local_hist = local_history_table.read(local_pc_idx);

        stall_stage();
        val<2> old_local_ctr = local_pht.read(old_local_hist);
        val<1> old_local_pred = counter_prediction(old_local_ctr);

        val<2> new_local_ctr = update_counter(old_local_ctr, taken);
        stall_stage();
        local_pht.write(old_local_hist, new_local_ctr);

        val<LOCAL_HISTORY_LEN> new_local_hist =
            (old_local_hist << hard<1>{}) | taken;
        stall_stage();
        local_history_table.write(local_pc_idx, new_local_hist);

        // ----- Global predictor -----
        val<GHR_LEN> global_idx = get_global_index(branch_pc);

        stall_stage();
        val<2> old_global_ctr = global_pht.read(global_idx);
        val<1> old_global_pred = counter_prediction(old_global_ctr);

        val<2> new_global_ctr = update_counter(old_global_ctr, taken);
        stall_stage();
        global_pht.write(global_idx, new_global_ctr);

        // ----- Chooser -----
        val<1> predictors_disagree = old_local_pred ^ old_global_pred;

        stall_stage();
        val<CHOOSER_PC_BITS> chooser_idx = get_chooser_index(branch_pc);
        val<2> old_chooser_ctr = chooser.read(chooser_idx);

        val<1> global_correct = ~(old_global_pred ^ taken);
        val<1> local_correct = ~(old_local_pred ^ taken);

        val<2> chooser_toward_global =
            select(old_chooser_ctr == val<2>{3}, val<2>{3}, val<2>{old_chooser_ctr + val<2>{1}});
        val<2> chooser_toward_local =
            select(old_chooser_ctr == val<2>{0}, val<2>{0}, val<2>{old_chooser_ctr - val<2>{1}});

        stall_stage();
        execute_if(predictors_disagree & global_correct, [&]() {
            chooser.write(chooser_idx, chooser_toward_global);
        });

        stall_stage();
        execute_if(predictors_disagree & local_correct, [&]() {
            chooser.write(chooser_idx, chooser_toward_local);
        });

        // ----- Global history -----
        ghr = (ghr << hard<1>{}) | taken;
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) override {}
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


template <class = void>
struct tournament : predictor {
    local_two_level<> local_predictor;
    global_pred<> global_predictor;

    reg<2> chooser; // 2 bit saturating counter

    reg<1> last_local;
    reg<1> last_global;

    tournament() {
        local_predictor.bind_parent(this);
        global_predictor.bind_parent(this);
    }

    val<1> predict(val<64> inst_pc) {
        val<1> local_pred = local_predictor.predict(inst_pc);
        val<1> global_pred = global_predictor.predict(inst_pc);
        last_local = local_pred;
        need_extra_cycle(1);
        last_global = global_pred;

        val<1> use_local = chooser >> 1;
        return select(use_local, local_pred, global_pred);
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc) {
        return hard<0>{};
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc) {
        return predict(inst_pc);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken,
                       [[maybe_unused]] val<64> next_pc) {
        local_predictor.update_condbr(branch_pc, taken, next_pc);
        global_predictor.update_condbr(branch_pc, taken, next_pc);

        val<1> local_correct = (last_local == taken);
        val<1> global_correct = (last_global == taken);

        val<1> favor_local = local_correct & ~global_correct;
        val<1> favor_global = ~local_correct & global_correct;

        execute_if(favor_local, [&] {
            chooser = select(chooser == 3, chooser, val<2>{chooser + 1});
            need_extra_cycle(1);
        });

        execute_if(favor_global, [&] {
            chooser = select(chooser == 0, chooser, val<2>{chooser - 1});
            need_extra_cycle(1);
        });
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
