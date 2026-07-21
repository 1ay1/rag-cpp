add_executable(ragcpp_cli main.cpp)
set_target_properties(ragcpp_cli PROPERTIES OUTPUT_NAME ragcpp)
target_link_libraries(ragcpp_cli PRIVATE ragcpp)
