#include "../cbp.hpp"
#include "../harcom.hpp"

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

struct tree_node {
    // leaves are predictor numbers
    // internal nodes are 2 bit saturating counters
    // if counter >= 2, go right, else go left
    reg<2> choice;
    reg<3> predictor_number;

    tree_node() : choice(0) {}

    tree_node(val<2> c) : choice(c) {}

    static tree_node leaf(val<3> p) {
        tree_node t;
        t.predictor_number = p;
        return t;
    }
};

template <class = void>
struct tree_chooser : predictor {
    static constexpr u64 NUM_CHOICES = 4;
    static constexpr u64 TREE_DEPTH = 2;
    static constexpr u64 TREE_SIZE = NUM_CHOICES | (NUM_CHOICES - 1);
    static constexpr u64 FIRST_LEAF = NUM_CHOICES;

    // 1 based indexing for tree
    std::array<tree_node, TREE_SIZE + 1> choice_tree;
    reg<1> last_p0;
    reg<1> last_p1;
    reg<1> last_p2;
    reg<1> last_p3;
    reg<1> last_final_pred;

    // Predictors
    tournament<>* predictor1;
    two_level<12, 14>* predictor2;
    two_level<14, 16>* predictor3;
    two_level<16, 20>* predictor4;

    tree_chooser()
    : choice_tree{{
          tree_node{0},
          tree_node{1},
          tree_node{1},
          tree_node{1},
          tree_node::leaf(0),
          tree_node::leaf(1),
          tree_node::leaf(2),
          tree_node::leaf(3)
      }},
      predictor1(new tournament<>()),
      predictor2(new two_level<12, 14>()),
      predictor3(new two_level<14, 16>()),
      predictor4(new two_level<16, 20>())
    {
        predictor1->bind_parent(this);
        predictor1->local_predictor.bind_parent(this);
        predictor1->global_predictor.bind_parent(this);
        predictor2->bind_parent(this);
        predictor3->bind_parent(this);
        predictor4->bind_parent(this);
    }

    val<1> predict(val<64> inst_pc) {
        last_p0 = predictor1->predict(inst_pc);
        need_extra_cycle(1);

        last_p1 = predictor2->predict(inst_pc);
        need_extra_cycle(1);

        last_p2 = predictor3->predict(inst_pc);
        need_extra_cycle(1);

        last_p3 = predictor4->predict(inst_pc);
        need_extra_cycle(1);

        val<1> root_left  = val<1>{choice_tree[1].choice <  val<2>{2}};
        val<1> root_right = val<1>{choice_tree[1].choice >= val<2>{2}};

        val<1> left_0 = val<1>{choice_tree[2].choice <  val<2>{2}};
        val<1> left_1 = val<1>{choice_tree[2].choice >= val<2>{2}};

        val<1> right_2 = val<1>{choice_tree[3].choice <  val<2>{2}};
        val<1> right_3 = val<1>{choice_tree[3].choice >= val<2>{2}};

        val<1> left_pred  = (left_0 & last_p0) | (left_1 & last_p1);
        val<1> right_pred = (right_2 & last_p2) | (right_3 & last_p3);

        last_final_pred = (root_left & left_pred) | (root_right & right_pred);
        need_extra_cycle(1);

        return last_final_pred;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        return last_final_pred;
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return last_final_pred;
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        return predict(inst_pc);
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        return predict(inst_pc);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken,
                       [[maybe_unused]] val<64> next_pc) {
        update_tree(taken);
        predictor1->update_condbr(branch_pc, taken, next_pc);
        predictor2->update_condbr(branch_pc, taken, next_pc);
        predictor3->update_condbr(branch_pc, taken, next_pc);
        predictor4->update_condbr(branch_pc, taken, next_pc);
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) { }

    val<1> is_predictor_correct(u64 predictor_num, val<1> taken) {
        if (predictor_num == 0) return val<1>{last_p0 == taken};
        if (predictor_num == 1) return val<1>{last_p1 == taken};
        if (predictor_num == 2) return val<1>{last_p2 == taken};
        return val<1>{last_p3 == taken};
    }

    void dec_counter(reg<2>& counter) {
        execute_if(val<1>{counter != val<2>{0}}, [&] {
            counter = counter - val<2>{1};
        });
    }

    void inc_counter(reg<2>& counter) {
        execute_if(val<1>{counter != val<2>{3}}, [&] {
            counter = counter + val<2>{1};
        });
    }

    void update_tree(val<1> taken_val) {
        val<1> p0 = is_predictor_correct(0, taken_val);
        val<1> p1 = is_predictor_correct(1, taken_val);
        val<1> p2 = is_predictor_correct(2, taken_val);
        val<1> p3 = is_predictor_correct(3, taken_val);

        val<1> np0 = val<1>{p0 == val<1>{0}};
        val<1> np1 = val<1>{p1 == val<1>{0}};
        val<1> np2 = val<1>{p2 == val<1>{0}};
        val<1> np3 = val<1>{p3 == val<1>{0}};

        val<1> left_0  = val<1>{choice_tree[2].choice <  val<2>{2}};
        val<1> left_1  = val<1>{choice_tree[2].choice >= val<2>{2}};
        val<1> right_2 = val<1>{choice_tree[3].choice <  val<2>{2}};
        val<1> right_3 = val<1>{choice_tree[3].choice >= val<2>{2}};

        val<1> left_correct  = (left_0 & p0) | (left_1 & p1);
        val<1> right_correct = (right_2 & p2) | (right_3 & p3);

        val<1> left_wrong  = val<1>{left_correct == val<1>{0}};
        val<1> right_wrong = val<1>{right_correct == val<1>{0}};

        execute_if(left_correct & right_wrong, [&] {
            dec_counter(choice_tree[1].choice);
        });
        need_extra_cycle(1);

        execute_if(left_wrong & right_correct, [&] {
            inc_counter(choice_tree[1].choice);
        });
        need_extra_cycle(1);

        execute_if(p0 & np1, [&] {
            dec_counter(choice_tree[2].choice);
        });
        need_extra_cycle(1);

        execute_if(np0 & p1, [&] {
            inc_counter(choice_tree[2].choice);
        });
        need_extra_cycle(1);

        execute_if(p2 & np3, [&] {
            dec_counter(choice_tree[3].choice);
        });
        need_extra_cycle(1);

        execute_if(np2 & p3, [&] {
            inc_counter(choice_tree[3].choice);
        });
        need_extra_cycle(1);
    }
};
