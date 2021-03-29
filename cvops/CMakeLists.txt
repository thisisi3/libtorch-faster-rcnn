
add_library(
	cvops STATIC
	cuda_helpers.h
	nms_cpu.cpp
	nms_cuda.cu
	nms.h
	ROIAlign_cpu.cpp
	ROIAlign_cuda.cu
	ROIAlign.h
	ROIPool_cpu.cpp
	ROIPool_cuda.cu
	ROIPool.h
	vision_cpu.h
	vision_cuda.h
)

target_include_directories(cvops PUBLIC ${PROJECT_SOURCE_DIR}/cvops)
target_link_libraries(cvops PUBLIC ${TORCH_LIBRARIES} Python3::Python)





