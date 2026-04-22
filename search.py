import random
import subprocess
from pathlib import Path

SPACE = {
    "LOGLB":  [6, 7],
    "NUMG":   [20, 24, 28], 
    "LOGG":   list(range(16, 23)), 
    "LOGB":   list(range(16, 23)), 
    "TAGW":   list(range(16, 22)),  
    "GHIST":  [1000, 2000, 4000], 
    "LOGP1":  list(range(16, 23)),  
    "GHIST1": list(range(12, 28)), 
}

PARAM_ORDER = [
    "LOGLB",
    "NUMG",
    "LOGG",
    "LOGB",
    "TAGW",
    "GHIST",
    "LOGP1",
    "GHIST1",
]

# Group related parameters together for crossover
PARAM_GROUPS = [
    ["LOGLB"],
    ["NUMG", "GHIST"],
    ["LOGG", "LOGB", "TAGW"],
    ["LOGP1", "GHIST1"],
]


RANDOM_SEED = 42
POPULATION_SIZE = 12  
MAX_GENERATIONS = 8 
ELITE_FRAC = 0.25
MUTATION_RATE = 0.70
RANDOM_INJECTION_FRAC = 0.12

MIN_GENERATIONS_BEFORE_STOP = 3
CONVERGENCE_PATIENCE = 3

LOCAL_SEARCH_ITERS = 30 

random.seed(RANDOM_SEED)

CACHE = {}  


def candidate_to_key(candidate):
    return tuple(candidate[k] for k in PARAM_ORDER)


def choose_candidate():
    return {k: random.choice(SPACE[k]) for k in PARAM_ORDER}


def choose_unique_candidate(seen):
    while True:
        cand = choose_candidate()
        key = candidate_to_key(cand)
        if key not in seen:
            seen.add(key)
            return cand

def generate_initial_population():
    population = []
    seen = set()

    while len(population) < POPULATION_SIZE:
        cand = choose_unique_candidate(seen)
        if cand is None:
            break
        population.append(cand)

    return population


def fitness(candidate):
    return evaluate(candidate)


def evaluate(candidate):
    key = candidate_to_key(candidate)
    if key in CACHE:
        return CACHE[key]

    loglb  = candidate["LOGLB"]
    numg   = candidate["NUMG"]
    logg   = candidate["LOGG"]
    logb   = candidate["LOGB"]
    tagw   = candidate["TAGW"]
    ghist  = candidate["GHIST"]
    logp1  = candidate["LOGP1"]
    ghist1 = candidate["GHIST1"]

    root = Path(__file__).resolve().parent
    trace = root / "gcc_test_trace.gz"
    predictor = f"tage<{loglb},{numg},{logg},{logb},{tagw},{ghist},{logp1},{ghist1}>"

    print("Candidate:", candidate)
    print("Predictor:", predictor)

    compile_cmd = [
        "g++",
        "-std=c++20",
        "cbp.cpp",
        "-O3",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-Wold-style-cast",
        "-Werror",
        "-Wno-deprecated-declarations",
        "-Wno-mismatched-tags",
        "-lz",
        f"-DPREDICTOR={predictor}",
        "-o",
        "cbp_eval",
    ]

    # Compile
    try:
        subprocess.run(
            compile_cmd,
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("COMPILATION FAILED")
        print("STDERR:\n", e.stderr)
        CACHE[key] = float("-inf")
        return CACHE[key]

    # Execute
    try:
        exe = root / "cbp_eval"
        result = subprocess.run(
            [str(exe), str(trace), "gcc_test", "1000000", "40000000"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("EXECUTION FAILED")
        print("STDERR:\n", e.stderr)
        CACHE[key] = float("-inf")
        return CACHE[key]

    # parse output
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        print("No output lines")
        CACHE[key] = float("-inf")
        return CACHE[key]

    fields = [x.strip() for x in rows[-1].split(",")]
    if len(fields) < 9:
        print("Not enough CSV fields")
        CACHE[key] = float("-inf")
        return CACHE[key]

    try:
        cond_branches = int(fields[3])
        p2_mispred = int(fields[8])
    except Exception as e:
        print("PARSE ERROR:", e)
        CACHE[key] = float("-inf")
        return CACHE[key]

    if cond_branches <= 0:
        print("ERROR: cond_branches <= 0")
        CACHE[key] = float("-inf")
        return CACHE[key]

    accuracy = 1.0 - (p2_mispred / cond_branches)

    print(f"\nFINAL ACCURACY: {accuracy:.8f}\n")

    CACHE[key] = accuracy
    return accuracy


def grouped_crossover(parent1, parent2):
    child = {}

    for group in PARAM_GROUPS:
        source_parent = random.choice([parent1, parent2])
        for key in group:
            child[key] = source_parent[key]

    return child


def mutate(candidate):
    mutated = candidate.copy()

    num_mutations = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    keys_to_mutate = random.sample(PARAM_ORDER, num_mutations)

    for key in keys_to_mutate:
        values = SPACE[key]
        current_val = mutated[key]
        current_idx = values.index(current_val)

        if random.random() < 0.75:
            possible_offsets = [-2, -1, 1, 2]
            offset = random.choice(possible_offsets)
            new_idx = max(0, min(current_idx + offset, len(values) - 1))
            mutated[key] = values[new_idx]
        else:
            mutated[key] = random.choice(values)

    return mutated


def tournament_select(population, tournament_size=4):
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=fitness)


def build_next_generation(current_gen):
    current_gen = sorted(current_gen, key=fitness, reverse=True)

    elite_size = max(1, int(POPULATION_SIZE * ELITE_FRAC))
    elites = current_gen[:elite_size]

    next_gen = elites[:]
    seen = {candidate_to_key(c) for c in next_gen}

    random_injections = max(1, int(POPULATION_SIZE * RANDOM_INJECTION_FRAC))
    injected_so_far = 0

    while len(next_gen) < POPULATION_SIZE:
        if injected_so_far < random_injections and random.random() < 0.12:
            cand = choose_unique_candidate(seen)
            if cand is not None:
                next_gen.append(cand)
                injected_so_far += 1
            continue

        p1 = tournament_select(current_gen, tournament_size=4)
        p2 = tournament_select(current_gen, tournament_size=4)

        child = grouped_crossover(p1, p2)

        if random.random() < MUTATION_RATE:
            child = mutate(child)

        key = candidate_to_key(child)
        if key not in seen:
            seen.add(key)
            next_gen.append(child)

    return next_gen


def local_search(best_candidate, max_iterations=LOCAL_SEARCH_ITERS):
    current_best = best_candidate.copy()
    current_score = fitness(current_best)

    for iteration in range(max_iterations):
        improved = False

        for key in PARAM_ORDER:
            current_val = current_best[key]
            values = SPACE[key]
            current_idx = values.index(current_val)

            for offset in [-1, 1]:
                new_idx = current_idx + offset
                if 0 <= new_idx < len(values):
                    variant = current_best.copy()
                    variant[key] = values[new_idx]
                    variant_score = fitness(variant)

                    if variant_score > current_score:
                        print(f"  Improvement: {current_score:.8f} -> {variant_score:.8f}")
                        print(f"    {key}: {current_val} -> {values[new_idx]}")
                        current_best = variant
                        current_score = variant_score
                        improved = True
                        break

            if improved:
                break

        if not improved:
            print(f"No improvement {iteration}")
            break

    print(f"Final accuracy: {current_score:.8f}\n")
    return current_best, current_score

def main():
    print(f"Using random seed: {RANDOM_SEED}")
    population = generate_initial_population()

    if not population:
        print("Failed to generate initial population.")
        return

    best_score = float("-inf")
    best_candidate = None
    no_improvement_count = 0

    for generation in range(MAX_GENERATIONS):
        population.sort(key=fitness, reverse=True)

        current_best = population[0]
        current_score = fitness(current_best)

        print(f"Generation {generation}: best accuracy = {current_score:.8f}")
        print(current_best)

        if current_score > best_score:
            best_score = current_score
            best_candidate = current_best.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if (
            generation >= MIN_GENERATIONS_BEFORE_STOP
            and no_improvement_count >= CONVERGENCE_PATIENCE
        ):
            print(f"\nGA converged after {generation} generations.")
            break

        population = build_next_generation(population)

    print("\n[GA RESULT]")
    print(f"Best GA accuracy: {best_score:.8f}")
    print(f"Best GA parameters: {best_candidate}")

    best_candidate, best_score = local_search(best_candidate)

    print("\n>>> FINAL RESULT <<<")
    print(f"Final accuracy: {best_score:.8f}")
    print(f"Final parameters: {best_candidate}")


if __name__ == "__main__":
    main()