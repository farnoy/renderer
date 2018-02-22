allocator:
	@clang++ --std=c++14 amd_alloc.cc -c -o amd_alloc.o -g
	@ar rcs libamd_alloc.a amd_alloc.o