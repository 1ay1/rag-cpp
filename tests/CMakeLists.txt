add_executable(ragcpp_tests test_main.cpp)
target_link_libraries(ragcpp_tests PRIVATE ragcpp)
add_test(NAME ragcpp_tests COMMAND ragcpp_tests)

# The monolithic fixture intentionally discards Result values while arranging
# corpora; assertions exercise the later search outcome. Keep that test idiom
# without flooding clean builds with hundreds of identical nodiscard warnings.
if(MSVC)
    target_compile_options(ragcpp_tests PRIVATE /wd4834)
else()
    target_compile_options(ragcpp_tests PRIVATE -Wno-unused-result)
endif()

add_executable(ragcpp_c_api_tests test_c_api.cpp)
target_link_libraries(ragcpp_c_api_tests PRIVATE ragcpp)
add_test(NAME ragcpp_c_api_tests COMMAND ragcpp_c_api_tests)

# RCP/1 front-end integration tests (only when the RCP SDK is wired in).
if(TARGET ragcpp::rcp)
    add_executable(ragcpp_rcp_tests test_rcp.cpp)
    target_link_libraries(ragcpp_rcp_tests PRIVATE ragcpp::rcp)
    add_test(NAME ragcpp_rcp_tests COMMAND ragcpp_rcp_tests)
endif()
