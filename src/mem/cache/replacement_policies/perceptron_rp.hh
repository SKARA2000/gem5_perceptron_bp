#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_PERCEPTRON_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_PERCEPTRON_RP_HH__

#include "base/sat_counter.hh"
#include "mem/cache/replacement_policies/tree_plru_rp.hh"

namespace gem5
{

struct PerceptronRPParams;

namespace replacement_policy
{

class Perceptron : public TreePLRU
{
protected:
    /** Perceptron-specific implementation of replacement data. */
    struct PerceptronReplData : public TreePLRUReplData
    {

        // Hashes of the Last retrieved features for this cache block entry
        std::vector<int8_t> lastFeatureHashes;

        // Last PC that accessed this cache block entry
        Addr lastPC;

        // Last prediction outcome for cache block entry
        // True -> Dead Block Prediction; False -> Live Block Prediction
        bool prediction;

        /**
         * Default constructor. Sets last touch tick to zero.
         */
        PerceptronReplData(const uint64_t index, std::shared_ptr<PLRUTree> tree)
            : TreePLRUReplData(index, tree), lastPC(0), prediction(false) {}
    };

    // Number of features used in the perceptron
    const int numFeatures;

    // Table size of each feature
    const int tableSize;

    // Perceptron Feature Hash Tables
    std::vector<std::vector<int8_t>> featureTables;

    // PC history for feature calculation
    std::vector<Addr> pcHistory;

    // Prediction Threshold
    const int8_t predictionThreshold;

    // Bypass Prediction Threshold
    // const int8_t bypassThreshold;

    // Training Threshold
    const int8_t trainingThreshold;

    // Calculate features
    std::vector<int8_t> calculateFeatures(Addr pc, Addr blockAddr) const;

    // Compute y_out
    int computePrediction(const std::vector<int8_t>& features, Addr pc) const;

    // Train perceptron
    void train(const std::vector<int8_t>& features, Addr pc, bool trainIncr);

public:
    typedef PerceptronRPParams Params;
    Perceptron(const Params &p);
    ~Perceptron() = default;

    /**
     * Invalidate replacement data to set it as the next probable victim.
     * Sets its last touch tick as the starting tick.
     *
     * @param replacement_data Replacement data to be invalidated.
     */
    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
                                                                    override;

    /**
     * Touch an entry to update its replacement data.
     * Sets its last touch tick as the current tick.
     *
     * @param replacement_data Replacement data to be touched.
     */
    void touch(const std::shared_ptr<ReplacementData>& replacement_data,
        const PacketPtr pkt) override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const
        override;

    /**
     * Reset replacement data. Used when an entry is inserted.
     * Sets its last touch tick as the current tick.
     *
     * @param replacement_data Replacement data to be reset.
     */
    void reset(const std::shared_ptr<ReplacementData>& replacement_data,
        const PacketPtr pkt) override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const
        override;

    /**
     * Find replacement victim using LRU timestamps.
     *
     * @param candidates Replacement candidates, selected by indexing policy.
     * @return Replacement entry to be replaced.
     */
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const
                                                                    override;

    /**
     * Instantiate a replacement data entry.
     *
     * @return A shared pointer to the new replacement data.
     */
    std::shared_ptr<ReplacementData> instantiateEntry() override;
    };
}
}

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_PERCEPTRON_RP_HH__