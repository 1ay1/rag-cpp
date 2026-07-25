add_executable(ragcpp_tests test_main.cpp)
target_link_libraries(ragcpp_tests PRIVATE ragcpp)
add_test(NAME ragcpp_tests COMMAND ragcpp_tests)

add_executable(ragcpp_c_api_tests test_c_api.cpp)
target_link_libraries(ragcpp_c_api_tests PRIVATE ragcpp)
add_test(NAME ragcpp_c_api_tests COMMAND ragcpp_c_api_tests)

# RCP/1 front-end integration tests (only when the RCP SDK is wired in).
if(TARGET ragcpp::rcp)
    add_executable(ragcpp_rcp_tests test_rcp.cpp)
    target_link_libraries(ragcpp_rcp_tests PRIVATE ragcpp::rcp)
    add_test(NAME ragcpp_rcp_tests COMMAND ragcpp_rcp_tests)
endif()
