#include "layer_impl.hh"

#pragma once

namespace tensor_array
{
    namespace layers
    {
		template <class T>
		class LayerHolder
		{
		private:
			std::shared_ptr<T> layer_ptr;
		public:
			static_assert(std::is_base_of_v<LayerImpl, T>, "Not base");

			template <class Derived_Class>
			LayerHolder(const LayerHolder<Derived_Class>&);

			template <class Derived_Class>
			LayerHolder(LayerHolder<Derived_Class>&&);

			template <typename ... Args>
			LayerHolder(Args&&...);

			LayerHolder(const std::shared_ptr<T>&);

			T* get() const;

			const std::shared_ptr<T>& get_shared() const;

			template <typename ... Args>
			auto operator()(Args...);

			T& operator*();
			T* operator->() const;
		};

		template<class T>
		template<typename ...Args>
		inline LayerHolder<T>::LayerHolder(Args&& ... args):
			layer_ptr(std::make_shared<T>(std::forward<Args>(args)...))
		{
		}

		template<class T>
		template<class Derived_Class>
		inline LayerHolder<T>::LayerHolder(const LayerHolder<Derived_Class>& layer_hold) :
			layer_ptr(layer_hold.get_shared())
		{
			static_assert(std::is_base_of_v<T, Derived_Class>, "Not derived from T");
		}

		template<class T>
		template<class Derived_Class>
		inline LayerHolder<T>::LayerHolder(LayerHolder<Derived_Class>&& layer_hold) :
			layer_ptr(std::move(layer_hold.get_shared()))
		{
			static_assert(std::is_base_of_v<T, Derived_Class>, "Not derived from T");
		}

		template<class T>
		template<typename ...Args>
		inline auto LayerHolder<T>::operator()(Args ... args)
		{
			if (!this->layer_ptr->is_running)
				this->layer_ptr->init_value(args...);
			this->layer_ptr->is_running = true;
			return this->layer_ptr->calculate(args...);
		}

		template<class T>
		inline LayerHolder<T>::LayerHolder(const std::shared_ptr<T>& layer_ptr):
			layer_ptr(layer_ptr)
		{
		}
		template<class T>
		inline T* LayerHolder<T>::get() const
		{
			return this->layer_ptr.get();
		}
		template<class T>
		inline const std::shared_ptr<T>& LayerHolder<T>::get_shared() const
		{
			return this->layer_ptr;
		}
		template<class T>
		inline T& LayerHolder<T>::operator*()
		{
			return this->layer_ptr.operator*();
		}
		template<class T>
		inline T* LayerHolder<T>::operator->() const
		{
			return this->layer_ptr.operator->();
		}
		;
	}
}
