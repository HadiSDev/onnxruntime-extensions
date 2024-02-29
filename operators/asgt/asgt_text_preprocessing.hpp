// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_tensor.h"
#include "ustring.h"
#include "opencv2/core.hpp"
#include <unordered_map>

struct KernelAsgtTextPreprocessing : BaseKernel {
  KernelAsgtTextPreprocessing(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& text,
               ortc::Tensor<int64>& input_ids) const;

private:
  std::unordered_map<std::u32string, int32_t> vocab_;
};

void KernelAsgtTextPreprocessing_Split(const std::u32string& text, std::vector<std::u32string>& words);

void KernelAsgtTextPreprocessing_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
                                        const std::vector<ustring>& texts,
                                        std::vector<int64_t>& indicies,
                                        int64_t& max_size);
