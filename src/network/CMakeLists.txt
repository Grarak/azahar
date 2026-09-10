# ENABLE_ROOM gates the multiplayer room protocol (and with it enet). The rest of this library
# is not multiplayer: artic_base loads titles off a real console, and socket_manager backs the
# guest's own SOC:U sockets, so both stay in either way. The headers stay compiled too - the
# packet and WifiPacket types they define are part of the NWM service's signatures - only the
# implementations that would drag in enet go.
add_library(network STATIC EXCLUDE_FROM_ALL
    artic_base/artic_base_client.cpp
    artic_base/artic_base_client.h
    artic_base/artic_base_common.h
    network.cpp
    network.h
    packet.cpp
    packet.h
    precompiled_headers.h
    socket_manager.cpp
    socket_manager.h
    $<$<BOOL:${ENABLE_ROOM}>:announce_multiplayer_session.cpp announce_multiplayer_session.h>
    $<$<BOOL:${ENABLE_ROOM}>:room.cpp room.h room_member.cpp room_member.h>
    $<$<BOOL:${ENABLE_ROOM}>:verify_user.cpp verify_user.h>
)

create_target_directory_groups(network)

if (ENABLE_WEB_SERVICE)
    target_link_libraries(network PRIVATE web_service)
endif()

target_link_libraries(network PRIVATE citra_common Boost::serialization httplib)
if (ENABLE_ROOM)
    target_link_libraries(network PRIVATE enet)
endif()

if (CITRA_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(network PRIVATE precompiled_headers.h)
endif()
