G0 = 9.80665

KSP_BODY_CONSTANTS = {
    'Sun': {
        'attractor'             : 'None',
        'mass'                  : 1.988475415966536e+30,
        'gravational_parameter' : 1.3271244004193939e+20,
        'rotational_period'     : 432000.0,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 696342000.0,
        'sphere_of_influence'   : None,
        'atmosphere_height'     : 600000.0,
        'angular_velocity'      : (-0.0, 0.0, 1.4544410433286079e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Earth': {
        'attractor'             : 'Sun',
        'mass'                  : 5.972365261370795e+24,
        'gravational_parameter' : 398600435436096.0,
        'rotational_period'     : 86164.098903691,
        'initial_rotation'      : 1.7485284405132353,
        'equatorial_radius'     : 6371000.0,
        'sphere_of_influence'   : 924649202.4610229,
        'atmosphere_height'     : 140000.0,
        'angular_velocity'      : (-0.0, 0.0, 7.292115146706924e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Mercury': {
        'attractor'             : 'Sun',
        'mass'                  : 3.301096181046679e+23,
        'gravational_parameter' : 22031780000000.02,
        'rotational_period'     : 5067031.68,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 2439700.0,
        'sphere_of_influence'   : 112408990.75442438,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 1.240013030109886e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Io': {
        'attractor'             : 'Jupiter',
        'mass'                  : 8.929943952440492e+22,
        'gravational_parameter' : 5959916033410.404,
        'rotational_period'     : 153042.32563796063,
        'initial_rotation'      : 3.3161255787892263,
        'equatorial_radius'     : 1811300.0,
        'sphere_of_influence'   : 7840344.603668602,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 4.1055213196662926e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Venus': {
        'attractor'             : 'Sun',
        'mass'                  : 4.867466257521636e+24,
        'gravational_parameter' : 324858592000000.06,
        'rotational_period'     : -20996797.016381,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 6049000.0,
        'sphere_of_influence'   : 616280853.7469519,
        'atmosphere_height'     : 145000.0,
        'angular_velocity'      : (0.0, 0.0, -2.992449420870076e-07),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Jupiter': {
        'attractor'             : 'Sun',
        'mass'                  : 1.8981872396165582e+27,
        'gravational_parameter' : 1.266865349218008e+17,
        'rotational_period'     : 35730.0,
        'initial_rotation'      : 0.4363323129985824,
        'equatorial_radius'     : 69373000.0,
        'sphere_of_influence'   : 48196176124.28713,
        'atmosphere_height'     : 1550000.0,
        'angular_velocity'      : (-0.0, 0.0, 0.00017585181380295513),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Titan': {
        'attractor'             : 'Saturn',
        'mass'                  : 1.3452248664299799e+23,
        'gravational_parameter' : 8978138376543.0,
        'rotational_period'     : 1378067.0375018918,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 2573300.0,
        'sphere_of_influence'   : 43324649.471155584,
        'atmosphere_height'     : 600000.0,
        'angular_velocity'      : (-0.0, 0.0, 4.559419198190466e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Phobos': {
        'attractor'             : 'Mars',
        'mass'                  : 1.0619510204993726e+16,
        'gravational_parameter' : 708754.6066894453,
        'rotational_period'     : 27574.878655452798,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 7250.0,
        'sphere_of_influence'   : 47000.0,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 0.0002278590374118334),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Vesta': {
        'attractor'             : 'Sun',
        'mass'                  : 2.590356269223623e+20,
        'gravational_parameter' : 17288244969.3,
        'rotational_period'     : 19231.2,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 262700.0,
        'sphere_of_influence'   : 39276806.019105665,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 0.00032671831748302683),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Moon': {
        'attractor'             : 'Earth',
        'mass'                  : 7.346031312426276e+22,
        'gravational_parameter' : 4902800066163.796,
        'rotational_period'     : 2370996.231427916,
        'initial_rotation'      : 0.4363323129985824,
        'equatorial_radius'     : 1737100.0,
        'sphere_of_influence'   : 66167158.6569544,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 2.650019103318263e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Mars': {
        'attractor'             : 'Sun',
        'mass'                  : 6.417120205436419e+23,
        'gravational_parameter' : 42828373620699.09,
        'rotational_period'     : 88642.6848,
        'initial_rotation'      : 0.4363323129985824,
        'equatorial_radius'     : 3375800.0,
        'sphere_of_influence'   : 577254070.8724953,
        'atmosphere_height'     : 125000.0,
        'angular_velocity'      : (-0.0, 0.0, 7.088216383964474e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Deimos': {
        'attractor'             : 'Mars',
        'mass'                  : 1440733351730922.2,
        'gravational_parameter' : 96155.69648120314,
        'rotational_period'     : 109082.14283454465,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 5456.0,
        'sphere_of_influence'   : 45000.0,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 5.760049393886492e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Ceres': {
        'attractor'             : 'Sun',
        'mass'                  : 9.384439503272361e+20,
        'gravational_parameter' : 62632500000.00001,
        'rotational_period'     : 32666.4,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 473000.0,
        'sphere_of_influence'   : 76962905.73054667,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 0.0001923439775175589),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Europa': {
        'attractor'             : 'Jupiter',
        'mass'                  : 4.798771928000401e+22,
        'gravational_parameter' : 3202738774922.8916,
        'rotational_period'     : 307004.4126746158,
        'initial_rotation'      : 4.014257279586958,
        'equatorial_radius'     : 1550800.0,
        'sphere_of_influence'   : 9727541.139687302,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 2.0466107481780512e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Ganymede': {
        'attractor'             : 'Jupiter',
        'mass'                  : 1.4815277091875051e+23,
        'gravational_parameter' : 9887834453334.145,
        'rotational_period'     : 618573.7139661426,
        'initial_rotation'      : 4.014257279586958,
        'equatorial_radius'     : 2624100.0,
        'sphere_of_influence'   : 24359376.755679477,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 1.0157536871221615e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Callisto': {
        'attractor'             : 'Jupiter',
        'mass'                  : 1.0756972288910635e+23,
        'gravational_parameter' : 7179289361397.27,
        'rotational_period'     : 1443348.1665247271,
        'initial_rotation'      : 3.3161255787892263,
        'equatorial_radius'     : 2409300.0,
        'sphere_of_influence'   : 37703185.401052766,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 4.3532014332398735e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Saturn': {
        'attractor'             : 'Sun',
        'mass'                  : 5.683361227113286e+26,
        'gravational_parameter' : 3.793120749865224e+16,
        'rotational_period'     : 38052.0,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 57216000.0,
        'sphere_of_influence'   : 54475312962.69387,
        'atmosphere_height'     : 2000000.0,
        'angular_velocity'      : (-0.0, 0.0, 0.0001651210266787445),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Mimas': {
        'attractor'             : 'Saturn',
        'mass'                  : 3.751114760386449e+19,
        'gravational_parameter' : 2503523999.9999995,
        'rotational_period'     : 81843.3641338282,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 198200.0,
        'sphere_of_influence'   : 396000.0,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 7.677085825682189e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Enceladus': {
        'attractor'             : 'Saturn',
        'mass'                  : 1.0805165904253472e+20,
        'gravational_parameter' : 7211454165.826001,
        'rotational_period'     : 118762.23017829933,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 252100.0,
        'sphere_of_influence'   : 488586.7425416514,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 5.290558536789479e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Tethys': {
        'attractor'             : 'Saturn',
        'mass'                  : 6.174795301586136e+20,
        'gravational_parameter' : 41211077826.41,
        'rotational_period'     : 163439.09662736705,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 531100.0,
        'sphere_of_influence'   : 1213919.4442904096,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 3.844358808165058e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Dione': {
        'attractor'             : 'Saturn',
        'mass'                  : 1.0955272709844651e+21,
        'gravational_parameter' : 73116366487.31999,
        'rotational_period'     : 236764.66679728994,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 561400.0,
        'sphere_of_influence'   : 1954777.8593541621,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 2.6537681454636308e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Rhea': {
        'attractor'             : 'Saturn',
        'mass'                  : 2.3065720571749214e+21,
        'gravational_parameter' : 153942464353.5,
        'rotational_period'     : 390535.27028361894,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 763800.0,
        'sphere_of_influence'   : 3675619.316702547,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 1.6088650079201657e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Iapetus': {
        'attractor'             : 'Saturn',
        'mass'                  : 1.8056734217045643e+21,
        'gravational_parameter' : 120512088703.29999,
        'rotational_period'     : 6853087.600913562,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 734500.0,
        'sphere_of_influence'   : 22505227.67654332,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 9.168400687512001e-07),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Uranus': {
        'attractor'             : 'Sun',
        'mass'                  : 8.681273407389496e+25,
        'gravational_parameter' : 5793951322279009.0,
        'rotational_period'     : 62063.712,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 24702000.0,
        'sphere_of_influence'   : 51692514225.02219,
        'atmosphere_height'     : 1400000.0,
        'angular_velocity'      : (-0.0, 0.0, 0.00010123766537166817),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Miranda': {
        'attractor'             : 'Uranus',
        'mass'                  : 6.47207839766994e+19,
        'gravational_parameter' : 4319516899.2321,
        'rotational_period'     : 122181.68386292394,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 235700.0,
        'sphere_of_influence'   : 459755.11887867434,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 5.142493627955491e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Ariel': {
        'attractor'             : 'Uranus',
        'mass'                  : 1.250561040888104e+21,
        'gravational_parameter' : 83463444317.70477,
        'rotational_period'     : 217797.41205049,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 578900.0,
        'sphere_of_influence'   : 2209608.9963886514,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 2.8848760175915277e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Umbriel': {
        'attractor'             : 'Uranus',
        'mass'                  : 1.2749829331517432e+21,
        'gravational_parameter' : 85093380944.89388,
        'rotational_period'     : 358093.2969770942,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 584700.0,
        'sphere_of_influence'   : 3101969.6038674996,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 1.7546224294674516e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Titania': {
        'attractor'             : 'Uranus',
        'mass'                  : 3.400374289402056e+21,
        'gravational_parameter' : 226943700374.12476,
        'rotational_period'     : 752245.0550777856,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 788900.0,
        'sphere_of_influence'   : 7532779.334918171,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 8.352577746795393e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Oberon': {
        'attractor'             : 'Uranus',
        'mass'                  : 3.076430463128435e+21,
        'gravational_parameter' : 205323430253.5623,
        'rotational_period'     : 1163272.6889030833,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 761400.0,
        'sphere_of_influence'   : 9677834.494020222,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 5.401300457852546e-06),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Neptune': {
        'attractor'             : 'Sun',
        'mass'                  : 1.0241260971459244e+26,
        'gravational_parameter' : 6835099502439672.0,
        'rotational_period'     : 58000.32,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 24085000.0,
        'sphere_of_influence'   : 86636358530.1769,
        'atmosphere_height'     : 1250000.0,
        'angular_velocity'      : (-0.0, 0.0, 0.00010833018347449784),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Triton': {
        'attractor'             : 'Neptune',
        'mass'                  : 2.139018622379465e+22,
        'gravational_parameter' : 1427598140725.034,
        'rotational_period'     : -507782.078471251,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 1353400.0,
        'sphere_of_influence'   : 11964318.522340473,
        'atmosphere_height'     : 110000.0,
        'angular_velocity'      : (0.0, 0.0, -1.2373783112031041e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Pluto': {
        'attractor'             : 'Sun',
        'mass'                  : 1.3029718219752756e+22,
        'gravational_parameter' : 869613817760.8748,
        'rotational_period'     : 551855.277413765,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 1187000.0,
        'sphere_of_influence'   : 3114585521.171567,
        'atmosphere_height'     : 110000.0,
        'angular_velocity'      : (-0.0, 0.0, 1.1385567130254402e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
    'Charon': {
        'attractor'             : 'Pluto',
        'mass'                  : 1.5864357163862e+21,
        'gravational_parameter' : 105879988860.1881,
        'rotational_period'     : 584486.1739413965,
        'initial_rotation'      : 0.0,
        'equatorial_radius'     : 603500.0,
        'sphere_of_influence'   : 8440471.516897446,
        'atmosphere_height'     : 0.0,
        'angular_velocity'      : (-0.0, 0.0, 1.0749929745659937e-05),
        'direction'             : (0.0, 0.0, 1.0)
    },
}
