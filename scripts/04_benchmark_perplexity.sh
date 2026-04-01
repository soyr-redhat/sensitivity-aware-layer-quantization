#!/bin/bash
# Run perplexity benchmark on all models
#
# Prerequisites:
#   - llama.cpp compiled with perplexity tool
#   - GGUF models (uniform and mixed-precision)
#   - Test dataset (will be auto-generated if not found)
#
# Usage:
#   1. Customize paths below
#   2. Run: ./04_benchmark_perplexity.sh

set -e

# CUSTOMIZE THESE PATHS
PERPLEXITY_TOOL="${PERPLEXITY_TOOL:-llama-perplexity}"  # or full path
MODEL_DIR="${MODEL_DIR:-models}"  # Directory containing GGUF models
TEST_FILE="${TEST_FILE:-data/perplexity_test.txt}"  # Test dataset
OUTPUT_DIR="${OUTPUT_DIR:-outputs/benchmarks}"  # Results directory

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$TEST_FILE")"

# Generate test file if it doesn't exist
if [ ! -f "$TEST_FILE" ]; then
    echo "Generating test file..."
    cat > "$TEST_FILE" <<'EOF'
= Valkyria Chronicles III =

Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit. Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " .

The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .

It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .

= = Gameplay = =

As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text . The player progresses through a series of linear missions , gradually unlocked as the story progresses . Mission objectives typically involve capturing the enemy 's camp or defeating a specific enemy unit . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty level . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .

The game 's battle system , called BLiTZ ( Battle of Live Tactical Zones ) , is carried over from previous entries . During missions , players select each unit using a top @-@ down perspective of the battlefield map : once a character is selected , the player moves the character around the battlefield in third @-@ person . Action points are used to move units on the battlefield , with different terrain types costing different amounts of action points to traverse . While moving , a unit can come under fire from enemy units , causing action points to drain . Units can take cover behind sandbags and other obstacles , and some terrain types allow characters to duck behind cover to avoid enemy fire . The number of times a particular unit can act depends on their class .

= The Eiffel Tower =

The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.

The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels. The climb from ground level to the first level is over 300 steps, as is the climb from the first level to the second. Although there is a staircase to the top level, it is usually accessible only by lift.
EOF
    # Repeat to get enough tokens (need 4096+ for context of 2048)
    for i in {1..4}; do
        cat "$TEST_FILE" >> "${TEST_FILE}.tmp"
    done
    mv "${TEST_FILE}.tmp" "$TEST_FILE"
    echo "Created test file with $(wc -w < "$TEST_FILE") words"
fi

echo "Model,Size_GB,Perplexity,Status" > "$OUTPUT_DIR/perplexity_results.csv"

run_perplexity() {
    local model_path="$1"
    local model_name="$2"

    echo ""
    echo "========================================"
    echo "Testing: $model_name"
    echo "========================================"

    if [ ! -f "$model_path" ]; then
        echo "ERROR: Model not found"
        echo "$model_name,0,N/A,NOT_FOUND" >> "$OUTPUT_DIR/perplexity_results.csv"
        return 1
    fi

    local size_gb=$(du -b "$model_path" | awk '{printf "%.2f", $1/1024/1024/1024}')

    local output=$($PERPLEXITY_TOOL -m "$model_path" -f "$TEST_FILE" -c 2048 -ngl 0 -t 8 2>&1)

    local ppl=$(echo "$output" | grep "Final estimate: PPL" | grep -oP '\d+\.\d+' | head -1)

    if [ -z "$ppl" ]; then
        echo "ERROR: Could not extract perplexity"
        echo "$model_name,$size_gb,FAILED,ERROR" >> "$OUTPUT_DIR/perplexity_results.csv"
        return 1
    fi

    echo "Result: PPL = $ppl"
    echo "$model_name,$size_gb,$ppl,OK" >> "$OUTPUT_DIR/perplexity_results.csv"

    # Save full output
    echo "$output" > "$OUTPUT_DIR/${model_name}_ppl_full.txt"
}

echo "================================================================================"
echo "PERPLEXITY BENCHMARK - All Models"
echo "================================================================================"

# Uniform baselines (adjust paths to match your model filenames)
run_perplexity "$MODEL_DIR/mistral-7b-q2k.gguf" "Q2_K"
run_perplexity "$MODEL_DIR/mistral-7b-q4km.gguf" "Q4_K_M"
run_perplexity "$MODEL_DIR/mistral-7b-q6k.gguf" "Q6_K"
run_perplexity "$MODEL_DIR/mistral-7b-q8.gguf" "Q8_0"

# Mixed-precision models
run_perplexity "$MODEL_DIR/mixed/mistral-7b-aggressive_mixed.gguf" "aggressive_mixed"
run_perplexity "$MODEL_DIR/mixed/mistral-7b-balanced_mixed.gguf" "balanced_mixed"
run_perplexity "$MODEL_DIR/mixed/mistral-7b-conservative_mixed.gguf" "conservative_mixed"

echo ""
echo "================================================================================"
echo "RESULTS"
echo "================================================================================"
echo ""
cat "$OUTPUT_DIR/perplexity_results.csv"
echo ""
echo "Full results saved to: $OUTPUT_DIR/perplexity_results.csv"
