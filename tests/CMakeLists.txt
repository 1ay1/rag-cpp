add_executable(ragcpp_tests test_main.cpp)
target_link_libraries(ragcpp_tests PRIVATE ragcpp)
add_test(NAME ragcpp_tests COMMAND ragcpp_tests)

add_executable(ragcpp_c_api_tests test_c_api.cpp)
target_link_libraries(ragcpp_c_api_tests PRIVATE ragcpp)
add_test(NAME ragcpp_c_api_tests COMMAND ragcpp_c_api_tests)
