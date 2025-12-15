"""
Tests for WebSocket functionality
"""
import pytest
import json


@pytest.mark.asyncio
class TestConnectionManager:
    """Tests for WebSocket ConnectionManager"""

    async def test_connect(self, ws_manager, mock_websocket):
        """Test connecting a websocket"""
        await ws_manager.connect(mock_websocket, "test-client")
        
        assert mock_websocket in ws_manager.active_connections
        assert mock_websocket in ws_manager.connection_metadata
        mock_websocket.accept.assert_called_once()

    def test_disconnect(self, ws_manager, mock_websocket):
        """Test disconnecting a websocket"""
        # First add the websocket
        ws_manager.active_connections.add(mock_websocket)
        ws_manager.connection_metadata[mock_websocket] = {
            "client_id": "test",
            "subscribed_feeds": set()
        }
        
        ws_manager.disconnect(mock_websocket)
        
        assert mock_websocket not in ws_manager.active_connections
        assert mock_websocket not in ws_manager.connection_metadata

    async def test_subscribe_to_feed(self, ws_manager, mock_websocket):
        """Test subscribing to a feed"""
        await ws_manager.connect(mock_websocket, "test-client")
        ws_manager.subscribe_to_feed(mock_websocket, "CCTV-001")
        
        assert "CCTV-001" in ws_manager.feed_subscribers
        assert mock_websocket in ws_manager.feed_subscribers["CCTV-001"]

    async def test_unsubscribe_from_feed(self, ws_manager, mock_websocket):
        """Test unsubscribing from a feed"""
        await ws_manager.connect(mock_websocket, "test-client")
        ws_manager.subscribe_to_feed(mock_websocket, "CCTV-001")
        ws_manager.unsubscribe_from_feed(mock_websocket, "CCTV-001")
        
        assert "CCTV-001" not in ws_manager.feed_subscribers

    async def test_broadcast(self, ws_manager, mock_websocket):
        """Test broadcasting message to all connections"""
        await ws_manager.connect(mock_websocket, "test-client")

        message = {"type": "test", "data": "hello"}
        await ws_manager.broadcast(message)

        # Messages are wrapped with compression metadata
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["compressed"] == False
        assert call_args["data"] == message

    async def test_send_detection_update(self, ws_manager, mock_websocket):
        """Test sending detection update to subscribers"""
        await ws_manager.connect(mock_websocket, "test-client")
        ws_manager.subscribe_to_feed(mock_websocket, "CCTV-001")

        detection_data = {"vehicle_count": 3}
        await ws_manager.send_detection_update("CCTV-001", detection_data)

        # Verify message was sent (wrapped with compression metadata)
        call_args = mock_websocket.send_json.call_args
        assert call_args is not None
        wrapper = call_args[0][0]
        assert wrapper["compressed"] == False
        sent_message = wrapper["data"]
        assert sent_message["type"] == "detection"
        assert sent_message["feed_id"] == "CCTV-001"

    async def test_send_heartbeat(self, ws_manager, mock_websocket):
        """Test sending heartbeat to all connections"""
        await ws_manager.connect(mock_websocket, "test-client")

        await ws_manager.send_heartbeat()

        # Messages are wrapped with compression metadata
        call_args = mock_websocket.send_json.call_args
        assert call_args is not None
        wrapper = call_args[0][0]
        assert wrapper["compressed"] == False
        sent_message = wrapper["data"]
        assert sent_message["type"] == "heartbeat"

    def test_get_stats(self, ws_manager, mock_websocket):
        """Test getting connection statistics"""
        ws_manager.active_connections.add(mock_websocket)
        
        stats = ws_manager.get_stats()
        
        assert stats["total_connections"] == 1
        assert "feed_subscriptions" in stats


@pytest.mark.asyncio
class TestWebSocketMessageHandling:
    """Tests for WebSocket message handling"""

    async def test_subscribe_action(self, ws_manager, mock_websocket):
        """Test handling subscribe action"""
        from websocket_manager import handle_websocket_messages
        
        await ws_manager.connect(mock_websocket, "test-client")
        
        # Set up mock to receive message then disconnect
        mock_websocket.receive_text.side_effect = [
            json.dumps({"action": "subscribe", "feed_id": "CCTV-001"}),
            Exception("Disconnected")
        ]
        
        try:
            await handle_websocket_messages(mock_websocket, ws_manager)
        except:
            pass
        
        # Verify subscription response was sent
        assert mock_websocket.send_json.called

    async def test_ping_action(self, ws_manager, mock_websocket):
        """Test handling ping action"""
        from websocket_manager import handle_websocket_messages

        await ws_manager.connect(mock_websocket, "test-client")

        mock_websocket.receive_text.side_effect = [
            json.dumps({"action": "ping"}),
            Exception("Disconnected")
        ]

        try:
            await handle_websocket_messages(mock_websocket, ws_manager)
        except:
            pass

        # Verify pong was sent (messages are wrapped with compression metadata)
        call_args_list = mock_websocket.send_json.call_args_list
        pong_sent = any(
            call[0][0].get("data", {}).get("type") == "pong"
            for call in call_args_list
        )
        assert pong_sent

    async def test_invalid_json(self, ws_manager, mock_websocket):
        """Test handling invalid JSON"""
        from websocket_manager import handle_websocket_messages

        await ws_manager.connect(mock_websocket, "test-client")

        mock_websocket.receive_text.side_effect = [
            "not valid json",
            Exception("Disconnected")
        ]

        try:
            await handle_websocket_messages(mock_websocket, ws_manager)
        except:
            pass

        # Verify error response was sent (messages are wrapped with compression metadata)
        call_args_list = mock_websocket.send_json.call_args_list
        error_sent = any(
            call[0][0].get("data", {}).get("type") == "error"
            for call in call_args_list
        )
        assert error_sent

