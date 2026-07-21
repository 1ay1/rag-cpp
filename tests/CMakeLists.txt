add_executable(ragcpp_tests test_main.cpp)
target_link_libraries(ragcpp_tests PRIVATE ragcpp)
add_test(NAME ragcpp_tests COMMAND ragcpp_tests)
