// Stubs — speaker embedding requires git-lfs data we don't have
#include "speaker-embedding-model.h"
#include "online-clusterer.h"

extern "C" {
    unsigned char speaker_embedding_model_ort_bytes[] = {0};
    size_t speaker_embedding_model_ort_byte_count = 0;
}

SpeakerEmbeddingModel::SpeakerEmbeddingModel(bool) : ort_api(nullptr), ort_env(nullptr),
    ort_session_options(nullptr), ort_memory_info(nullptr), embedding_session(nullptr), log_ort_run(false) {}
SpeakerEmbeddingModel::~SpeakerEmbeddingModel() {}
int SpeakerEmbeddingModel::load_from_memory(const uint8_t*, size_t) { return 0; }
int SpeakerEmbeddingModel::calculate_embedding(const float*, size_t, std::vector<float>*) { return 0; }

OnlineClusterer::OnlineClusterer(const OnlineClustererOptions& o) : options(o) {}
OnlineClusterer::~OnlineClusterer() {}
uint64_t OnlineClusterer::embed_and_cluster(const std::vector<float>&, float) { return 0; }
