#include "../cbp.hpp"
#include "../harcom.hpp"
#include "predictors.hpp"

using namespace hcm;

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
