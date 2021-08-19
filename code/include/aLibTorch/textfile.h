#include "torchtype.h"

namespace a::lt {

/**
 * @brief tensor를 file_name으로 저장
 * 
 * @param tensor     최대 2 dimension까지. float type
 * @param file_name  저장할 path
 */
void save_tensor_as_text(Tensor tensor, std::string file_name);

}