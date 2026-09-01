// Owns the one compiled precompiled-header object for the boost-multiprecision carrier
// (bcc32_reference.hpp / bcc32_coordinate.hpp). Every other host-only executable target
// reuses this compiled PCH via target_precompile_headers(... REUSE_FROM bcc32_pch) instead
// of each compiling its own copy -- see the PCH block at the end of CMakeLists.txt.
