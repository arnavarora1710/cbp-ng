#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

union tree_node {
    val<2> choice;
    val<3> predictor_number;
};

struct tree_chooser {
    static constexpr u64 NUM_CHOICES = 8;
    static constexpr u64 TREE_DEPTH = 3;
    static constexpr u64 TREE_SIZE = NUM_CHOICES | (NUM_CHOICES - 1);
    static constexpr u64 FIRST_LEAF = NUM_CHOICES - 1;

    ram<tree_node, TREE_SIZE> choice_tree;

    val<3> get_predictor_number() {
        u64 node_index = 0;
        for (u64 i = 0; i < TREE_DEPTH; i++) {
            tree_node node = choice_tree[node_index];
            if (node.choice == 0)
                node_index = (node_index << 1) + 1; // Go left
            else
                node_index = (node_index << 1) + 2; // Go right
        }
        return choice_tree[node_index].predictor_number;
    }

    void update_tree(bool taken) {
    }
};
