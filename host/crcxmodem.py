# Algorithm adapted to python from https://www.nongnu.org/avr-libc/user-manual/group__util__crc.html


def crc_xmodem_update(crc, data):
    crc = (crc ^ (data << 8)) & 0xFFFF
    for _ in range(8):
        if crc & 0x8000:
            crc = (crc << 1) ^ 0x1021
        else:
            crc <<= 1
    return crc & 0xFFFF


def calc_crc(msg: bytes) -> bytes:
    """Calculates the CRC16/XMODEM checksum for provided data

    :param msg: A list of chars to calculate the checksum over
    :type msg: bytes
    :return: 16 bit checksum calculated as a 32 bit bytes object
    :rtype: bytes
    """
    x = 0
    for uchar in msg:
        x = crc_xmodem_update(x, uchar)
    return x.to_bytes(4, "little")
