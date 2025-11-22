# Future plans

## Backend

### Support different backends

1. Memory backend: all tensor data is in memory

2. SHM (Shared memory) backend: all tensor data is in shared memory, can be used for cross-process sharing on the same node

3. Local FS backend: uses mmap to synchronize with the underlying file

4. GPFS backend: uses user space page cache to synchronize with the underlying file

### Requirements

1. When `open(uri, mode)` opens a tensor at path uri, automatically select backend based on the file system type where uri is located.

2. When uri is under /dev/shm, use SHM backend

3. When uri is on GPFS, use GPFS backend

4. When uri is on local disk, use Local FS backend
