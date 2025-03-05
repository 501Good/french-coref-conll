from ancor_utils import WordIndex

TO_CORRECT = {
    "u-MENTION-gpascault_1349681076425": {"to": WordIndex.from_string("#s1.u3.w9")},
    "u-MENTION-gpascault_1349680953025": {"to": WordIndex.from_string("#s1.u3.w9")},
    "u-MENTION-agoudjo_1330424956021": {"to": WordIndex.from_string("#s1.u1.w3")},
    "u-MENTION-agoudjo_1330424940640": {"to": WordIndex.from_string("#s1.u1.w3")},
    "u-MENTION-adrouin_1333311898676": {"to": WordIndex.from_string("#s1.u6.w30")},
    "u-MENTION-jmuzerelle_1361266912065": {"to": WordIndex.from_string("#s4.u19.w38")},
    "u-MENTION-sduchon_1343638142721": {"to": WordIndex.from_string("#s1.u1.w13")},
    "u-MENTION-sduchon_1343637957300": {"to": WordIndex.from_string("#s1.u1.w13")},
    "u-MENTION-jmuzerelle_1351086275515": {"to": WordIndex.from_string("#s18.u63.w9")},
    "u-MENTION-sduchon_1342946896111": {"to": WordIndex.from_string("#s1.u1.w11")},
    "u-MENTION-sduchon_1342946917806": {"to": WordIndex.from_string("#s1.u1.w11")},
    "u-MENTION-jmuzerelle_1353665145590": {"to": WordIndex.from_string("#s4.u1.w44")},
    "u-MENTION-jmuzerelle_1353935608198": {"to": WordIndex.from_string("#s16.u11.w24")},
    "u-MENTION-jmuzerelle_1372751868353": {"to": WordIndex.from_string("#s4.u1.w44")},
}

TO_REMOVE = {
    "u-MENTION-agoudjo_1330424998014",
    "u-MENTION-jmuzerelle_1364976050720",
    "u-MENTION-jmuzerelle_1364976058894",
    "u-MENTION-adrouin_1333311968915",
    "u-MENTION-jmuzerelle_1360591343483",
    "u-MENTION-jmuzerelle_1360591330488",
    "u-MENTION-jmuzerelle_1361283022127",
    "u-MENTION-jmuzerelle_1361266935262",
    "u-MENTION-jmuzerelle_1361283013298",
    "u-MENTION-jmuzerelle_1371129334073",
    "u-MENTION-jmuzerelle_1364467875536",
    "u-MENTION-jmuzerelle_1351086286529",
    "u-MENTION-jmuzerelle_1351086251943",
    "u-MENTION-sduchon_1342946935137",
    "u-MENTION-jmuzerelle_1353935616747",
    "u-MENTION-jmuzerelle_1353679970752",
    "u-MENTION-jmuzerelle_1354610946754",
    "u-MENTION-jmuzerelle_1354611012799",
    "u-MENTION-jmuzerelle_1354611460592",
    "u-MENTION-jmuzerelle_1354609716261",
    "u-MENTION-jmuzerelle_1354610935254",
    "u-MENTION-jmuzerelle_1354611451818",
    "u-MENTION-sduchon_1330028568172",
    "u-MENTION-sduchon_1330028558988",
}


def apply_patch(mentions: dict[str, dict]) -> None:
    for k, v in TO_CORRECT.items():
        if k in mentions:
            mentions[k].update(v)

    for i in TO_REMOVE:
        mentions.pop(i, None)
