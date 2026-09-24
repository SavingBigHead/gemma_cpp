package("highway_gemma")
set_homepage("https://github.com/google/highway")
set_description("Performance-portable, length-agnostic SIMD with runtime dispatch")
set_license("Apache-2.0")

add_urls("https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz")
add_versions("1.4.0", "e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c")

add_configs("contrib", { description = "Build SIMD-related utilities", default = true, type = "boolean" })
add_deps("cmake")

on_install(function(package)
	io.replace(
		"CMakeLists.txt",
		"    hwy/robust_statistics.h\n",
		"    hwy/robust_statistics.h\n    hwy/stats.h\n",
		{ plain = true }
	)
	io.replace("CMakeLists.txt", "    hwy/profiler.cc\n", "    hwy/profiler.cc\n    hwy/stats.cc\n", { plain = true })

	local configs = {
		"-DHWY_ENABLE_INSTALL=ON",
		"-DBUILD_TESTING=OFF",
		"-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"),
		"-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"),
		"-DHWY_ENABLE_CONTRIB=" .. (package:config("contrib") and "ON" or "OFF"),
	}
	import("package.tools.cmake").install(package, configs)
end)
package_end()

set_project("gemma")
set_languages("c++17")
set_encodings("utf-8")

add_requires("highway_gemma 1.4.0", { configs = { contrib = true }, system = false })
add_requires("sentencepiece v0.2.1", { configs = { shared = false } })
add_requires("nlohmann_json v3.12.0")
add_requires("cpp-httplib v0.18.1", { configs = { ssl = true } })
add_requires("benchmark v1.9.5")
add_requires("gtest v1.17.0")

option("build-dll")
set_default(false)
set_showmenu(true)
set_description("Build the Gemma shared library and C API")
option_end()

option("enable-tests")
set_default(false)
set_showmenu(true)
set_description("Build Gemma tests")
option_end()

local library_sources = {
	"compression/compress.cc",
	"evals/benchmark_helper.cc",
	"evals/cross_entropy.cc",
	"gemma/attention.cc",
	"gemma/configs.cc",
	"gemma/flash_attention.cc",
	"gemma/gemma.cc",
	"gemma/kv_cache.cc",
	"gemma/model_store.cc",
	"gemma/tensor_info.cc",
	"gemma/tokenizer.cc",
	"gemma/vit.cc",
	"gemma/weights.cc",
	"io/blob_store.cc",
	"io/fields.cc",
	"io/io_win.cc",
	"io/io.cc",
	"ops/matmul.cc",
	"ops/matmul_static_bf16.cc",
	"ops/matmul_static_f32.cc",
	"ops/matmul_static_i8.cc",
	"ops/matmul_static_nuq.cc",
	"ops/matmul_static_sfp.cc",
	"paligemma/image.cc",
	"util/allocator.cc",
	"util/basics.cc",
	"util/mat.cc",
	"util/threading.cc",
	"util/threading_context.cc",
	"util/topology.cc",
	"util/zones.cc",
}

local function configure_gemma_target()
	add_files(library_sources)
	add_includedirs(".", { public = true })
	add_packages("highway_gemma", "sentencepiece", { public = true })
	if is_plat("windows") then
		add_defines("_CRT_SECURE_NO_WARNINGS", "NOMINMAX")
		add_cxxflags("-Wno-deprecated-declarations")
	end
end

target("libgemma")
set_kind("static")
set_basename("gemma")
configure_gemma_target()
target_end()

if has_config("build-dll") then
	target("gemma_shared")
	set_kind("shared")
	set_basename("gemma")
	add_defines("GEMMA_EXPORTS")
	configure_gemma_target()
	add_files("gemma/bindings/context.cc", "gemma/bindings/c_api.cc")
	add_installfiles("gemma/bindings/c_api.h", "gemma/bindings/GemmaInterop.cs", { prefixdir = "include/gemma" })
	target_end()
end

target("gemma")
set_kind("binary")
add_files("gemma/run.cc")
add_deps("libgemma")
target_end()

target("single_benchmark")
set_kind("binary")
add_files("evals/benchmark.cc")
add_deps("libgemma")
add_packages("nlohmann_json")
target_end()

target("benchmarks")
set_kind("binary")
set_default(false)
add_files("evals/benchmarks.cc")
add_deps("libgemma")
add_packages("nlohmann_json", "benchmark")
target_end()

target("debug_prompt")
set_kind("binary")
add_files("evals/debug_prompt.cc")
add_deps("libgemma")
add_packages("nlohmann_json")
target_end()

target("migrate_weights")
set_kind("binary")
add_files("io/migrate_weights.cc")
add_deps("libgemma")
target_end()

target("gemma_api_server")
set_kind("binary")
add_files("gemma/api_server.cc")
add_deps("libgemma")
add_packages("nlohmann_json", "cpp-httplib")
target_end()

target("gemma_api_client")
set_kind("binary")
add_files("gemma/api_client.cc")
add_deps("libgemma")
add_packages("nlohmann_json", "cpp-httplib")
target_end()

target("hello_world")
set_kind("binary")
set_default(false)
add_files("examples/hello_world/run.cc")
add_deps("libgemma")
target_end()

target("simplified_gemma")
set_kind("binary")
set_default(false)
add_files("examples/simplified_gemma/run.cc")
add_deps("libgemma")
target_end()

if has_config("enable-tests") then
	local test_sources = {
		"compression/compress_test.cc",
		"compression/distortion_test.cc",
		"compression/nuq_test.cc",
		"compression/sfp_test.cc",
		"evals/gemma_test.cc",
		"gemma/flash_attention_test.cc",
		"gemma/tensor_info_test.cc",
		"io/blob_store_test.cc",
		"io/fields_test.cc",
		"ops/bench_matmul.cc",
		"ops/dot_test.cc",
		"ops/matmul_test.cc",
		"ops/ops_test.cc",
		"paligemma/image_test.cc",
		"paligemma/paligemma_test.cc",
		"util/basics_test.cc",
		"util/threading_test.cc",
	}

	for _, test_source in ipairs(test_sources) do
		target(path.basename(test_source))
		set_kind("binary")
		set_default(false)
		add_files(test_source)
		add_deps("libgemma")
		add_packages("gtest")
		add_defines("HWY_IS_TEST=1")
		target_end()
	end

	target("gemma_batch_bench")
	set_kind("binary")
	set_default(false)
	add_files("evals/gemma_batch_bench.cc")
	add_deps("libgemma")
	add_packages("gtest", "nlohmann_json")
	target_end()
end
