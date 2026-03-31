"""Async FreeSWITCH ESL client for call bridging/transfer.

The call is already on FreeSWITCH (comes in via WebSocket media stream).
When escalation happens, we use ESL to re-route that existing call to a
human agent — no new connection to establish, just a command to FreeSWITCH.

Two transfer modes
──────────────────
1. uuid_transfer  (blind transfer — default)
   FreeSWITCH takes the caller's channel (by UUID) and sends it through the
   dialplan to the destination extension/DID.  Bot's WebSocket drops
   immediately.  Caller hears ringing then connects to the human agent.

   ESL command:
       api uuid_transfer <uuid> <destination> XML <context>

2. originate + bridge  (warm transfer)
   FreeSWITCH first calls the human agent.  When the agent picks up,
   FreeSWITCH bridges them with the caller.  Bot's media stream ends once
   the bridge is established.  Useful when you want the agent to be ready
   before the caller is connected.

   ESL command:
       api originate {origination_caller_id_number=<caller_id>}user/<agent_ext> &bridge(<uuid>)

Environment variables (all optional):
  FREESWITCH_HOST                   — ESL host            (default: 127.0.0.1)
  FREESWITCH_PORT                   — ESL port            (default: 8021)
  FREESWITCH_PASSWORD               — ESL password        (default: ClueCon)
  FREESWITCH_TRANSFER_DESTINATION   — extension or DID    (default: 1000)
  FREESWITCH_CONTEXT                — dialplan context    (default: default)
"""

import asyncio
import os

from loguru import logger


class FreeSwitchESL:
    """Minimal async FreeSWITCH ESL client."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
    ):
        self.host = host or os.getenv("FREESWITCH_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("FREESWITCH_PORT", "8021"))
        self.password = password or os.getenv("FREESWITCH_PASSWORD", "ClueCon")

    # ── Low-level helpers ──────────────────────────────────────────────────

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=5.0,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach FreeSWITCH ESL at {self.host}:{self.port}: {exc}"
            ) from exc
        return reader, writer

    async def _authenticate(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Receive: Content-Type: auth/request
        challenge = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        if b"auth/request" not in challenge:
            raise ConnectionError(f"Unexpected ESL greeting: {challenge!r}")

        writer.write(f"auth {self.password}\n\n".encode())
        await writer.drain()

        reply = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        if b"+OK accepted" not in reply:
            raise PermissionError(f"ESL auth failed: {reply.decode(errors='replace')!r}")

    async def _send_api(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        command: str,
    ) -> str:
        writer.write(f"api {command}\n\n".encode())
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=10.0)
        return response.decode("utf-8", errors="replace").strip()

    async def _run(self, command: str) -> str:
        """Connect, authenticate, run one API command, disconnect."""
        logger.info(f"FreeSWITCH ESL | api {command}")
        reader, writer = await self._connect()
        try:
            await self._authenticate(reader, writer)
            result = await self._send_api(reader, writer, command)
            logger.info(f"FreeSWITCH ESL | result: {result!r}")
            return result
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────

    async def transfer_call(
        self,
        uuid: str,
        destination: str | None = None,
        context: str | None = None,
    ) -> str:
        """Blind transfer — move caller's existing channel to destination.

        The caller's FreeSWITCH channel (uuid) is sent through the dialplan
        to 'destination'.  The bot's WebSocket media stream is terminated.
        This is the simplest way to hand off to a human agent.

        Args:
            uuid:        FreeSWITCH call Unique-ID of the caller's channel.
            destination: Extension, DID, or dialplan destination.
                         Falls back to FREESWITCH_TRANSFER_DESTINATION env var.
            context:     FreeSWITCH dialplan context.
                         Falls back to FREESWITCH_CONTEXT env var or "default".

        Returns:
            "+OK" on success, "-ERR ..." on failure.
        """
        dest = destination or os.getenv("FREESWITCH_TRANSFER_DESTINATION", "1000")
        ctx = context or os.getenv("FREESWITCH_CONTEXT", "default")
        return await self._run(f"uuid_transfer {uuid} {dest} XML {ctx}")

    async def bridge_to_agent(
        self,
        caller_uuid: str,
        agent_destination: str | None = None,
        caller_id: str | None = None,
    ) -> str:
        """Warm transfer — originate a call to the agent then bridge with caller.

        FreeSWITCH calls the agent first.  When the agent picks up,
        the caller's channel is bridged to them.  The bot's media ends
        once the bridge is established.

        Args:
            caller_uuid:       FreeSWITCH UUID of the caller's channel.
            agent_destination: 'user/1000', a SIP URI, or a DID.
                               Falls back to FREESWITCH_TRANSFER_DESTINATION.
            caller_id:         Caller ID shown to the agent.
                               Falls back to FREESWITCH_CALLER_ID env var or
                               "Unknown".

        Returns:
            "+OK <new_uuid>" on success, "-ERR ..." on failure.
        """
        dest = agent_destination or os.getenv("FREESWITCH_TRANSFER_DESTINATION", "1000")
        cid = caller_id or os.getenv("FREESWITCH_CALLER_ID", "Unknown")

        # Prefix with 'user/' if destination looks like a plain extension
        if not dest.startswith("user/") and not dest.startswith("sofia/") and not dest.startswith("{"):
            dest = f"user/{dest}"

        cmd = f"originate {{origination_caller_id_number={cid}}}{dest} &bridge({caller_uuid})"
        return await self._run(cmd)
