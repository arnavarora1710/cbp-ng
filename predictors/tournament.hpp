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