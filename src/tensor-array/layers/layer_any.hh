/*
Copyright 2024 TensorArray-Creators

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "layer_holder.hh"

#pragma once

namespace tensor_array
{
    namespace layers
    {
        class AnyLayer
        {
        private:
            /* data */
            struct WrapLayerImpl
            {
                template<typename ...Args>
                auto operator()(Args&& ... args);
            };

            template <class T>
            struct LayerHolderWrap final : public WrapLayerImpl
            {
                LayerHolder<T> t;
                LayerHolderWrap();
                template<typename ...Args>
                auto operator()(Args&& ... args);
            };

            std::shared_ptr<WrapLayerImpl> wrap_ptr;
        public:
            template <class T>
            AnyLayer(LayerHolder<T>&&);
            template<typename ...Args>
            auto operator()(Args&& ... args);
            const std::shared_ptr<LayerImpl>& get_shared() const;
        };
        
        template <class T>
        AnyLayer::AnyLayer(LayerHolder<T>&& holder):
            wrap_ptr(std::make_shared<LayerHolderWrap<T>>(std::forward<LayerHolder<T>>(holder)))
        {
        }
        
        template<class T>
        template<typename ...Args>
        inline auto AnyLayer::LayerHolderWrap<T>::operator()(Args&& ... args)
        {
            return this->t(std::forward<Args>(args)...);
        }

        template<typename ...Args>
        inline auto AnyLayer::operator()(Args&& ... args)
        {
            return this->wrap_ptr->operator()(std::forward<Args>(args)...);
        }
	}
}
