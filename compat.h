// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_
#define THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_

#include "nccl_net.h"

typedef ncclNetProperties_v4_t ncclNetProperties_v3_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*pci_path)(int dev, char** path);
  ncclResult_t (*ptr_support)(int dev, int* supportedTypes);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
} ncclNet_v2_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*get_properties)(int dev, ncclNetProperties_v3_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*reg_mr)(void* comm, void* data, int size, int type,
                         void** mhandle);
  ncclResult_t (*dereg_mr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle,
                        void** request);
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close_send)(void* sendComm);
  ncclResult_t (*close_recv)(void* recvComm);
  ncclResult_t (*close_listen)(void* listenComm);
} ncclNet_v3_t;

#endif  // THIRD_PARTY_GPUS_NCCL_FASTSOCKET_PLUGIN_COMPAT_H_
