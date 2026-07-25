add_executable(ragcpp_cli main.cpp)
set_target_properties(ragcpp_cli PROPERTIES OUTPUT_NAME ragcpp)

# Link the RCP front-end when it was configured, so `ragcpp serve` can bring up
# a conformant RCP/1 endpoint over stdio or HTTP with no extra binary. Falls
# back to the core library (serve prints a build-hint) when RCP is off.
if(TARGET ragcpp::rcp)
    target_link_libraries(ragcpp_cli PRIVATE ragcpp::rcp)
else()
    target_link_libraries(ragcpp_cli PRIVATE ragcpp)
endif()
