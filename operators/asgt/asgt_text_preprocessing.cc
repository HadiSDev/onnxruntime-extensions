#include "asgt_text_preprocessing.hpp"
#include "nlohmann/json.hpp"
#include "text/string_functions.h"
#include <re2/re2.h>
#include <opencv2/core.hpp>
#include <dlib/matrix.h>

KernelAsgtTextPreprocessing::KernelAsgtTextPreprocessing(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  std::string vocab_as_string = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab");

  std::unordered_map<std::string, int32_t> vocab_map;
  auto parsed = nlohmann::json::parse(vocab_as_string);
  parsed.get_to(vocab_map);

  for (auto it = vocab_map.begin(); it != vocab_map.end(); ++it) {
    vocab_[ustring(it->first)] = it->second;
  }
}


void KernelAsgtTextPreprocessing_Split(const std::u32string& text, std::vector<std::u32string>& words) {
  std::string text_ = std::string(text.begin(), text.end());
  for (auto& c : text_) {
    c = ToLower(c);
  }
  re2::StringPiece lines_to(" ");
  re2::RE2 lines_pattern("[\t\n]");
  re2::RE2::Replace(&text_, lines_pattern, lines_to);

  re2::StringPiece special_chars_to("");
  re2::RE2 special_chars_pattern("([!\"\\#$%&()*+,-.\\/:;<=>?@\\[\\]^_`{|}])");
  re2::RE2::Replace(&text_, special_chars_pattern, special_chars_to);

  re2::StringPiece numbers_to("x");
  re2::RE2 numbers_pattern("[0-9]+");
  re2::RE2::Replace(&text_, numbers_pattern, numbers_to);

  re2::StringPiece empty_to("emptystring");
  re2::RE2 empty_pattern("^\\s*$");
  re2::RE2::Replace(&text_, empty_pattern, empty_to);

  std::string tmp;
  std::stringstream ss(text_);

  while(getline(ss, tmp, ' ')){
    words.emplace_back(tmp.begin(), tmp.end());
  }
}

void KernelAsgtTextPreprocessing_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
                                        const std::vector<ustring>& texts,
                                        std::vector<int64_t>& indicies,
                                        int64_t& max_size) {
  std::vector<int64_t> sizes;
  std::vector<std::vector<int64>> tmp;
  for(auto& t : texts) {
    std::vector<std::u32string> token_list;
    KernelAsgtTextPreprocessing_Split(t, token_list);
    std::vector<int64> token_ids;
    for(auto& c : token_list) {
      if(vocab.find(c) == vocab.end()) {
        int64 token_id = vocab.at(ustring("[UNK]"));
        token_ids.push_back(token_id);
      }
      else {
        int64 token_id = vocab.at(c);
        token_ids.push_back(token_id);
      }
    }
    tmp.push_back(token_ids);
    sizes.push_back(static_cast<int64_t>(token_ids.size()));
  }

  max_size = *max_element(sizes.begin(), sizes.end());

  for(auto& ids : tmp) {
    if(ids.size() < max_size) {
      for(int i = 0; i < (max_size - ids.size()); i++) {
        ids.push_back(vocab.at(ustring("[PAD]")));
      }
    }
  }
  for(auto& t : tmp) {
    for(auto& id : t) {
      indicies.push_back(id);
    }
  }
}

void KernelAsgtTextPreprocessing::Compute(const ortc::Tensor<std::string>& text,
                                          ortc::Tensor<int64>& input_ids) const {
  std::vector<ustring> str_input;
  for (auto& str : text.Data()) {
    str_input.emplace_back(str);
  }
  int64_t max_size;
  std::vector<int64_t> output_indicies;
  KernelAsgtTextPreprocessing_Tokenizer(vocab_, str_input, output_indicies, max_size);
  const std::vector<int64_t> outer_dims{text.NumberOfElement(), max_size};

  auto* p_out = input_ids.Allocate(outer_dims);
  std::copy(output_indicies.begin(), output_indicies.end(), p_out);
}