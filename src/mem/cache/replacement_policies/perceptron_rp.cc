#include "mem/cache/replacement_policies/perceptron_rp.hh"

#include <cassert>
#include <memory>

#include "sim/cur_tick.hh"
#include "base/logging.hh" // For fatal_if
#include "params/PerceptronRP.hh"

namespace gem5
{
namespace replacement_policy
{
Perceptron::Perceptron(const Params &p)
    :   TreePLRU(p),
        numFeatures(p.num_features),
        tableSize(p.table_size),
        featureTables(numFeatures, std::vector<int8_t>(tableSize, 0)),
        pcHistory(p.pc_history_length, 0),
        predictionThreshold(p.prediction_threshold),
        trainingThreshold(p.training_threshold) {}

std::vector<int8_t>
Perceptron::calculateFeatures(Addr pc, Addr blockAddr) const
{
    std::vector<int8_t> features(numFeatures, 0);
    features[0] = ((pc >> 2) % tableSize);
    features[1] = ((pcHistory[0] >> 1) % tableSize);
    features[2] = ((pcHistory[1] >> 2) % tableSize);
    features[3] = ((pcHistory[2] >> 3) % tableSize);
    features[4] = ((blockAddr >> 4) % tableSize);
    features[5] = ((blockAddr >> 7) % tableSize);

    return features;
}

int
Perceptron::computePrediction(const std::vector<int8_t>& features, Addr pc) const
{
    int y_out = 0;
    for (int i = 0; i < numFeatures; i++) {
        int8_t featureIndex = (features[i] ^ pc) % tableSize;
        y_out += featureTables[i][featureIndex];
    }
    return y_out;
}

void
Perceptron::train(const std::vector<int8_t>& features, Addr pc, bool trainIncr)
{
    const int8_t MAX_WEIGHT = 31;
    const int8_t MIN_WEIGHT = -32;
    for(int i = 0; i < numFeatures; i++) {
        int8_t featureIndex = (features[i] ^ pc) % tableSize;
        // Saturate weights between MIN_WEIGHT and MAX_WEIGHT
        if (trainIncr) {
            if (featureTables[i][featureIndex] < MAX_WEIGHT) {
                featureTables[i][featureIndex]++;
            }
        } else {
            if (featureTables[i][featureIndex] > MIN_WEIGHT) {
                featureTables[i][featureIndex]--;
            }
        }
    }
}

void
Perceptron::invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
{
    std::shared_ptr<PerceptronReplData> data =
        std::static_pointer_cast<PerceptronReplData>(replacement_data);

    // Cache eviction means that the block entry needs to be updated and
    // the predictor entry trained for no reuse
    if(!data->lastFeatureHashes.empty()) {
        // Cache block entry has a predictor entry associated, train it
        int y_out = computePrediction(data->lastFeatureHashes, data->lastPC);
        if((y_out < predictionThreshold) || (std::abs(y_out) < trainingThreshold)) {
            // Predictor was not correct or weakly correct, need to update the counters
            train(data->lastFeatureHashes, data->lastPC, true);
        }
    }
    // Cache block evicted. Reset metadata so that we don't get the wrong victim
    data->prediction = false;
    data->lastFeatureHashes.clear();
    data->lastPC = 0;

    // Invalidate using Tree-PLRU mechanism
    TreePLRU::invalidate(replacement_data);
}

void
Perceptron::touch(const std::shared_ptr<ReplacementData>& replacement_data, const PacketPtr pkt)
{
    std::shared_ptr<PerceptronReplData> data =
        std::static_pointer_cast<PerceptronReplData>(replacement_data);

    // Cache Hit means that the block entry needs to be updated if the
    // outcome is more than the training threshold
    Addr pc = pkt->req->hasPC() ? pkt->req->getPC() : 0;
    Addr blockAddr = pkt->getAddr(); // Assuming 64B block size

    if(!data->lastFeatureHashes.empty()) {
        int y_out = computePrediction(data->lastFeatureHashes, data->lastPC);
        if(y_out > -trainingThreshold) {
            // Predictor was not correct enough. Need to update the counters
            train(data->lastFeatureHashes, data->lastPC, false);
        }
    }

    // Update PC history
    for(int i = 2; i > 0; i--)
        pcHistory[i] = pcHistory[i - 1];
    pcHistory[0] = pc;

    // Update the last features in the replacement data
    data->lastFeatureHashes = calculateFeatures(pc, blockAddr);
    data->lastPC = pc;

    // Store new predction
    data->prediction = (computePrediction(data->lastFeatureHashes, pc) > predictionThreshold);

    TreePLRU::touch(replacement_data);
}

void
Perceptron::touch(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    TreePLRU::touch(replacement_data);
}

void
Perceptron::reset(const std::shared_ptr<ReplacementData>& replacement_data, const PacketPtr pkt)
{
    std::shared_ptr<PerceptronReplData> data =
        std::static_pointer_cast<PerceptronReplData>(replacement_data);

    Addr pc = pkt->req->hasPC() ? pkt->req->getPC() : 0;
    Addr blockAddr = pkt->getAddr() >> 6; // Assuming 64B block size

    // Store the lasst features
    data->lastFeatureHashes = calculateFeatures(pc, blockAddr);
    data->lastPC = pc;

    // Store new prediction
    data->prediction = (computePrediction(data->lastFeatureHashes, pc) > predictionThreshold);

    // Update PC history for replacement data
    for(int i = 2; i > 0; i--)
        pcHistory[i] = pcHistory[i - 1];
    pcHistory[0] = pc;

    TreePLRU::reset(replacement_data);
}

void
Perceptron::reset(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    TreePLRU::reset(replacement_data);
}

ReplaceableEntry*
Perceptron::getVictim(const ReplacementCandidates& candidates) const
{
    // Search for victim via branch prediction first
    for(const auto& candidate : candidates) {
        if(std::static_pointer_cast<PerceptronReplData>(
        candidate->replacementData)->prediction) {
            return candidate;
        }
    }

    // No victim found. Fallback to Tree-PLRU mechanism
    return TreePLRU::getVictim(candidates);
}

std::shared_ptr<ReplacementData>
Perceptron::instantiateEntry()
{
    // Generate a tree instance every numLeaves created
    if (count % numLeaves == 0) {
        treeInstance = new PLRUTree(numLeaves - 1, false);
    }

    // Create replacement data using current tree instance
    TreePLRUReplData* treePLRUReplData = new PerceptronReplData(
        (count % numLeaves) + numLeaves - 1,
        std::shared_ptr<PLRUTree>(treeInstance));

    // Update instance counter
    count++;

    return std::shared_ptr<ReplacementData>(treePLRUReplData);
}

}
}