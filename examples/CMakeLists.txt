cmake_minimum_required(VERSION 3.15)

# Example C++ programs
if(BUILD_EXAMPLES)
    add_executable(example_basic
        basic_example.cpp
    )
    
    target_link_libraries(example_basic PRIVATE dragon_tensor)
    
    target_include_directories(example_basic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
endif()

