
## Remote Procedure Call

A popular way to design distributed applications is by means of remote procedure calls. gRPC is
Google's remote procedure calls. You can read about remote procedure calls and gRPC here
[gRPC: Getting Started](https://github.com/grpc/grpc-common/blob/master/README.md) and here
[gRPC Basics: Go](https://github.com/grpc/grpc-go/blob/master/examples/gotutorial.md)

It would be useful to read about protocol buffers, Google's way of serializing data. 
[Protocal Buffers: Developer Guide](https://developers.google.com/protocol-buffers/docs/overview)

**Run the following commands to install the necessary libraries.**
```
go get -u github.com/golang/protobuf/{proto,protoc-gen-go}

go get -u google.golang.org/grpc
```

There is an example program in grpc called route_guide. Run the server and the client to test your 
installation. Open two terminals. In one run the following commands:
```
cd $GOPATH/src/google.golang.org/grpc/examples/route_guide/

go run server/server.go
```

If the server runs successfully, you should not see any output on this terminal. If you get an error, ask your teaching staff.

In another terminal window, type the following to start a client that will connect to the server that you just started:
```
cd $GOPATH/src/google.golang.org/grpc/examples/route_guide/

go run client/client.go
```

If the client runs successfully, you should see a lot of output similar to this:
```
...
2015/08/10 15:16:10 location:<latitude:405002031 longitude:-748407866 > 
2015/08/10 15:16:10 location:<latitude:409532885 longitude:-742200683 > 
2015/08/10 15:16:10 location:<latitude:416851321 longitude:-742674555 > 
2015/08/10 15:16:10 name:"3387 Richmond Terrace, Staten Island, NY 10303, USA" location:<latitude:406411633 longitude:-741722051 > 
2015/08/10 15:16:10 name:"261 Van Sickle Road, Goshen, NY 10924, USA" location:<latitude:413069058 longitude:-744597778 > 
2015/08/10 15:16:10 location:<latitude:418465462 longitude:-746859398 > 
2015/08/10 15:16:10 location:<latitude:411733222 longitude:-744228360 > 
2015/08/10 15:16:10 name:"3 Hasta Way, Newton, NJ 07860, USA" location:<latitude:410248224 longitude:-747127767 > 
2015/08/10 15:16:10 Traversing 74 points.
2015/08/10 15:16:10 Route summary: point_count:74 distance:720194535 
2015/08/10 15:16:10 Got message First message at point(0, 1)
2015/08/10 15:16:10 Got message Fourth message at point(0, 1)
2015/08/10 15:16:10 Got message First message at point(0, 1)
...
```

You may want to study the code for the `route_guide` example to try to understand what is going on.

# Building your own RPC-based key-value storage

RPC key-value store: The repository contains code for a very simple key-value storage
service, where the keys and values are strings. It offers the method Insert, which inserts a
key-value pair. It returns a bool indicating success or failure.

**Please look at, but do not change, the kv.pb.go file. This file contains important APIs and message
definitions needed for these exercises.**


```
type InsertRequest struct {
	Key   string `protobuf:"bytes,1,opt,name=key" json:"key,omitempty"`
	Value string `protobuf:"bytes,2,opt,name=value" json:"value,omitempty"`
}

type InsertResponse struct {
	Success bool `protobuf:"varint,1,opt,name=success" json:"success,omitempty"`
}

type LookupRequest struct {
	Key string `protobuf:"bytes,1,opt,name=key" json:"key,omitempty"`
}

type LookupResponse struct {
	Value string `protobuf:"bytes,1,opt,name=value" json:"value,omitempty"`
}

type KeysRequest struct {
}

type KeysResponse struct {
	Keys []string `protobuf:"bytes,1,rep,name=keys" json:"keys,omitempty"`
}

// Client API for KeyValueService service

type KeyValueServiceClient interface {
	Insert(ctx context.Context, in *InsertRequest, opts ...grpc.CallOption) (*InsertResponse, error)
	Lookup(ctx context.Context, in *LookupRequest, opts ...grpc.CallOption) (*LookupResponse, error)
	Keys(ctx context.Context, in *KeysRequest, opts ...grpc.CallOption) (*KeysResponse, error)
}

// Server API for KeyValueService service

type KeyValueServiceServer interface {
	Insert(context.Context, *InsertRequest) (*InsertResponse, error)
	Lookup(context.Context, *LookupRequest) (*LookupResponse, error)
	Keys(context.Context, *KeysRequest) (*KeysResponse, error)
}
```

**Tasks:**

1. Create a client and a connection to the server.

2. In the client, call the Insert() gRPC for a number of key/value pairs.

3. In the server, implement the Lookup() gRPC, which should return the value of the requested key.

4. In the server, implement the Keys() gRPC, which should return a slice of the keys 
(not the values) of the map back to the client.

5. In the client, call the Lookup() gRPC for each of the key/value pairs inserted and verify 
the result returned from the Lookup() gRPC matches the value inserted for the corresponding key.

6. In the client, call the Keys() gRPC and verify that the number of keys returned matches 
the expected number.

7. Several clients connecting to the server may read and write concurrently from the shared
key-value map. This will eventually cause inconsistencies in the map, unless some form of
protection is instituted. Implement locking at the appropriate locations
in the code. See [pkg/sync](http://golang.org/pkg/sync/).

Extras:

  - Explain why the clients can access the map at the server concurrently.
  - If you run your server without protection on the map, are you able to provoke inconsistencies in the map.

Troubleshooting if you get compile errors related to `kv.pb.go`, it may help to recompile the proto file:
```
cd lab2/grpc/proto
protoc --go_out=plugins=grpc:. kv.proto
```
