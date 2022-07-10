// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "compat.h"

#include "nccl_net.h"

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclFastSocketInit(ncclDebugLogger_t logFunction);
ncclResult_t ncclFastSocketDevices(int* ndev);
ncclResult_t ncclFastSocketPciPath(int dev, char** pciPath);
template <typename T>
ncclResult_t ncclFastSocketGetProperties(int dev, T* props);
ncclResult_t ncclFastSocketListen(int dev, void* opaqueHandle,
                                  void** listenComm);
ncclResult_t ncclFastSocketConnect(int dev, void* opaqueHandle,
                                   void** sendComm);
ncclResult_t ncclFastSocketAccept(void* listenComm, void** recvComm);
ncclResult_t ncclFastSocketRegMr(void* comm, void* data, int size, int type,
                                 void** mhandle);
ncclResult_t ncclFastSocketDeregMr(void* comm, void* mhandle);
ncclResult_t ncclFastSocketIsend_v2(void* sendComm, void* data, int size,
                                    void* mhandle, void** request);
ncclResult_t ncclFastSocketIrecv_v2(void* recvComm, void* data, int size,
                                    void* mhandle, void** request);
ncclResult_t ncclFastSocketIflush_v4(void* recvComm, void* data, int size,
                                     void* mhandle, void** request);
ncclResult_t ncclFastSocketTest(void* request, int* done, int* size);
ncclResult_t ncclFastSocketClose(void* opaqueComm);
ncclResult_t ncclFastSocketCloseListen(void* opaqueComm);

ncclResult_t ncclFastSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

ncclResult_t ncclFastSocketFlush(void* recvComm, void* data, int size,
                                 void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

volatile ncclNet_v2_t ncclNetPlugin_v2 = {
    "FastSocket",          ncclFastSocketInit,       ncclFastSocketDevices,
    ncclFastSocketPciPath, ncclFastSocketPtrSupport, ncclFastSocketListen,
    ncclFastSocketConnect, ncclFastSocketAccept,     ncclFastSocketRegMr,
    ncclFastSocketDeregMr, ncclFastSocketIsend_v2,   ncclFastSocketIrecv_v2,
    ncclFastSocketFlush,   ncclFastSocketTest,       ncclFastSocketClose,
    ncclFastSocketClose,   ncclFastSocketCloseListen};

volatile ncclNet_v3_t ncclNetPlugin_v3 = {
    "FastSocket",           ncclFastSocketInit,
    ncclFastSocketDevices,  ncclFastSocketGetProperties<ncclNetProperties_v3_t>,
    ncclFastSocketListen,   ncclFastSocketConnect,
    ncclFastSocketAccept,   ncclFastSocketRegMr,
    ncclFastSocketDeregMr,  ncclFastSocketIsend_v2,
    ncclFastSocketIrecv_v2, ncclFastSocketFlush,
    ncclFastSocketTest,     ncclFastSocketClose,
    ncclFastSocketClose,    ncclFastSocketCloseListen};
