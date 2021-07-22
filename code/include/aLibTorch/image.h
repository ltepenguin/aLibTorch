#pragma once
#include "torchtype.h"

namespace a::lt {

/**
 * @brief               image_resize 사용하세요.
 * 
 * @param   tensor      channel x height x width.
 * @param   w_scale     width scale parameter
 * @param   h_scale     height scale parameter
 * @return  Tensor      scaled tensor image
 */
Tensor image_scale(Tensor tensor, int w_scale, int h_scale);

/**
 * @brief               resize image. sampling strategy: nearest
 * 
 * @param   tensor      channel x height x width.
 * @param   width       new width
 * @param   height      new height
 * @return  Tensor      channel x new_width x new_height
 */
Tensor image_resize(Tensor tensor, int width, int height);

/**
 * @brief               tensor로부터 image 데이터로 생성
 *                      모든 image data는 Image class를 통해서 create 하고 Image class를 통해서 delete.
 *                      Image class는 name을 key 값으로 사용.
 *                      만약 같은 name을 갖는 image가 있을 경우, 기존 image 데이터 삭제.
 * 
 * @param   name        생성할 image 데이터 이름. Image class 에서 key 값으로 사용.
 * @param   tensor      channel x height x width. 값은 0 ~ 255 로 가정.
 * @return              tensor에 해당하는 이미지 데이터.
 */
agl::Image::spData create_image(Tensor tensor, std::string name);

/**
 * @brief               tensor로부터 texture 데이터 생성
 *                      name 이름의 image 데이터도 생성
 *                      만약 같은 name을 갖는 texture가 있을 경우, 기존 데이터 삭제.
 * 
 *                      example) create texture
 *                      create_texture(tensor, "my_texture", 512, 512, true)
 * 
 *                      example) render texture
 *                      agl::Render::plane()->texture("my_texture")->draw()
 *                      
 * 
 * @param   tensor      channel x height x width. 값은 0 ~ 255 로 가정.
 * @param   name        texture 이름. Image와 TextureLoader class 에서 key 값으로 사용.
 * @param   width       반드시 2의 배수: 8, 16, 32, 64, 128 ....
 * @param   height      반드시 2의 배수: 8, 16, 32, 64, 128 ....
 * @param   gl_nearest  texture rendering 시 blend 혹은 nearest 사용할지 결정.
 */
agl::Texture create_texture(Tensor tensor, std::string name, int width, int height, bool gl_nearest = false);

}