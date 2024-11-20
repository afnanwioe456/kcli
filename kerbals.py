import krpc

VANILLA_CREW_MEMBERS = [
    'Macdo',
    'Jan',
    'Panty',
    'Jesrey',
    'Erbo',
    'Herbal',
    'Gwenlans',
    'Haloly',
]

CUSTOM_CREW_MEMBERS = []


class Kerbal:
    def __init__(self, name, crew_member):
        self.name = name
        self.crew_member = crew_member

    @staticmethod
    def get_available_kerbals_name(count=1):
        with krpc.connect('kerbals',
                        address='127.0.0.1',
                        rpc_port=65534,
                        stream_port=65535) as conn:
            sc = conn.space_center
            if sc is None:
                return
            available_kerbals_name = []
            for k in VANILLA_CREW_MEMBERS:
                crew_member = sc.get_kerbal(k)
                if crew_member is None:
                    continue
                if not crew_member.on_mission:
                    available_kerbals_name.append(k)
                if len(available_kerbals_name) == count:
                    break
        return available_kerbals_name

    @staticmethod
    def _new_custom_kerbal_check(name):
        if name in CUSTOM_CREW_MEMBERS or len(VANILLA_CREW_MEMBERS) > 0:
            return True
        return False

    @staticmethod
    def new_custom_kerbal(name, conn):
        if not Kerbal._new_custom_kerbal_check(name):
            print('可用空闲成员不足')
            return
        available_kerbals = Kerbal.get_available_kerbals_name()
        if available_kerbals is None:
            return
        original_name = available_kerbals[0]
        crew_member = conn.space_center.get_kerbal(f'{original_name} Kerman')
        VANILLA_CREW_MEMBERS.remove(original_name)
        CUSTOM_CREW_MEMBERS.append(original_name)

        crew_member.name = name
        return crew_member

    @staticmethod
    def get_crew_members(name_list, conn, count=None):
        """返回name_list对应的CrewMember对象列表crew_member_list.
        当传入人数count > len(name_list)时，用可用成员补齐"""
        if count and len(name_list) < count:
            name_list += Kerbal.get_available_kerbals_name(count - len(name_list))
        crew_member_list = []
        for n in name_list:
            if n in VANILLA_CREW_MEMBERS or n in CUSTOM_CREW_MEMBERS:
                crew_member_list.append(conn.space_center.get_kerbal(f'{n} Kerman'))
            else:
                custom_crew_member = Kerbal.new_custom_kerbal(n, conn)
                if custom_crew_member:
                    crew_member_list.append(custom_crew_member)
        return crew_member_list



