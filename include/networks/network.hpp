#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/script.h>

#include <iostream>
#include <string>
namespace networks{
    class network_t
    {
    public:
    
        network_t(std::string network_weights_path){
            // initialize network
            if(network_weights_path == ""){
                std::cout <<"empty network_wieght_path, using default" << std::endl;
            }
            network_torch_module_ptr.reset(new torch::jit::script::Module(
                torch::jit::load(network_weights_path)));
        }

        virtual ~network_t(){
            network_torch_module_ptr.reset();
        }

        virtual at::Tensor forward(std::vector<torch::jit::IValue> mpnet_input_container) = 0;

    protected:
        /**
         * @brief loaded neural network
         */
        std::shared_ptr<torch::jit::script::Module> network_torch_module_ptr;

    };

}
#endif
