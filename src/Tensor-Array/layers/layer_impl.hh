#include <string>
#include <unordered_map>
#include "core/tensor.hh"
#include <utility>
#pragma once

#ifdef __WIN32__
#ifdef CUDA_ML_EXPORTS
#define CUDA_ML_API __declspec(dllexport)
#else
#define CUDA_ML_API __declspec(dllimport)
#endif
#else
#define CUDA_ML_API
#endif

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API LayerImpl
        {
        private:
            bool is_running = false;
            template <class T>
            friend class LayerHolder;
        protected:
            std::unordered_map<std::string, value::Tensor*> map_tensor;
            std::unordered_map<std::string, std::shared_ptr<LayerImpl>> map_layer;
        public:
            virtual ~LayerImpl() = default;
            virtual void update_weight(float = 0.001f) final;
            virtual void load_data(const std::string&) final;
            virtual void save_data(const std::string&) const final;
            inline LayerImpl() = default;
            constexpr LayerImpl(const LayerImpl&) = delete;
            constexpr LayerImpl(LayerImpl&&) = delete;
            inline LayerImpl operator=(const LayerImpl&) = delete;
            inline LayerImpl operator=(LayerImpl&&) = delete;
        };

        template <typename Return, typename ... Args>
        struct CalculateStruct
        {
            template <class T>
            friend class LayerHolder;
            virtual ~CalculateStruct() = default;
            virtual void init_value(Args ...) {}
            virtual Return calculate(Args ...) = 0;
        };

        struct TensorCalculateLayerImpl :
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&>
        {};
    }
}