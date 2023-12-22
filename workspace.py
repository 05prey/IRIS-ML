import hvlruleutil
from logger import logger
from commonkeys_v2 import *
from collections import deque


class CollectInput():
    def __init__(self, environment):
        self.myGlobalID = None
        self.myRuleHelper = None
        self.entityInstanceListObject = None
        self.input_space = {}
        self.env = environment
        self.entityInstanceListObject = None

        self.none_range = self.env.effective_ranges.none_range
        self.laser_range = self.env.effective_ranges.laser_range
        self.ar_range = self.env.effective_ranges.ar_range
        self.ir_range = self.env.effective_ranges.ir_range
        self.sar_range = self.env.effective_ranges.sar_range
        self.pr_range = self.env.effective_ranges.pr_range
        self.eo_range = self.env.effective_ranges.eo_range
        self.gun_range = self.env.effective_ranges.gun_range

        self.fired_history = {}
        self.my_message_history = deque([None]*10, maxlen=10)

    def _resetEntityList(self):
        self.entityInstanceListObject = hvlruleutil.EntityInstanceList()
        self.myRuleHelper.getAllEntityInstances(self.entityInstanceListObject)

    def captureGeneralInfo(self):
        try:
            self.input_space[GENERAL] = {}
            self.input_space[GENERAL].update({ID: self.myGlobalID,
                                              DAMAGE: self.myRuleHelper.getMyDamageStatus(),
                                              DPL: self.myRuleHelper.getMyCurrentDPLPercentage(),
                                              MYLASTDAMAGERID: self.myRuleHelper.getMyLastDamagerEntityGlobalId(),
                                              MYLASTDAMAGERNAME: self.myRuleHelper.getEntityName(self.myRuleHelper.getMyLastDamagerEntityGlobalId()),
                                              ISFIRED: self.env.isFired,
                                              CHAFFRELEASEFLAG: self.env.chaff_release_flag,
                                              FLARERELEASEFLAG: self.env.flare_release_flag,
                                              RUNAWAYDIRECTION: self.env.runAwayDirection,
                                              TRACKTIMER: self.env.track_counter})
        except Exception as E:
            logger().saveError(E, self.env.deployment_name)

    def captureMotionInfo(self):
        try:
            self.input_space[MOTION] = {}
            relative_bearings, bearings, distances = self.captureRelatives()
            self.input_space[MOTION].update({MYSPEED: self.myRuleHelper.getMySpeed(),
                                             MYHEADING: self.myRuleHelper.getMyHeading(),
                                             MSLALTITUDE: self.myRuleHelper.getMyMSLAltitude(),
                                             AGLALTITUDE: self.myRuleHelper.getMyAGLAltitude(),
                                             TERRAINALTITUDE: self.myRuleHelper.getTerrainAltitudeInMeters(),
                                             ECEF: [self.myRuleHelper.getMyLocationXYZ().x, self.myRuleHelper.getMyLocationXYZ().y, self.myRuleHelper.getMyLocationXYZ().z],
                                             LLA: [self.myRuleHelper.getMyLocation().latitude, self.myRuleHelper.getMyLocation().longitude],
                                             RELATIVEBEARING: relative_bearings,
                                             BEARING: bearings,
                                             DISTANCE: distances})
        except Exception as E:
            logger().savetxt(E, "BecomingInputErrorForMotion")

    def captureMunitionInfo(self):
        try:
            self.input_space[MUNITION] = {}
            self.input_space[MUNITION].update({MISSILECOUNT: self.myRuleHelper.getMissileCount(),
                                               ROCKETCOUNT: self.myRuleHelper.getRocketCount(),
                                               BULLETCOUNT: self.myRuleHelper.getBulletCount(),
                                               IRFLARECOUNT: self.myRuleHelper.getIRFlareCount(),
                                               ILLUMINATIONFLARECOUNT: self.myRuleHelper.getIlluminationFlareCount(),
                                               LASERGUIDEDMISSILECOUNT: self.myRuleHelper.getLaserGuidedMissileCount(),
                                               RADARGUIDEDMISSILECOUNT: self.myRuleHelper.getRadarGuidedMissileCount(),
                                               IRGUIDEDMISSILECOUNT: self.myRuleHelper.getIRGuidedMissileCount(),
                                               EOGUIDEDMISSILECOUNT: self.myRuleHelper.getEOGuidedMissileCount(),
                                               PROJECTILESHOTBULLETCOUNT: self.myRuleHelper.getProjectileShotBulletCount(),
                                               FLATSHOTBULLETCOUNT: self.myRuleHelper.getFlatShotBulletCount(),
                                               CHAFFCOUNT: self.myRuleHelper.getChaffCount(),
                                               SMOKECOUNT: self.myRuleHelper.getSmokeCount(),
                                               ACTIVERADARGUIDEDMISSILECOUNT: self.myRuleHelper.getActiveRadarGuidedMissileCount(),
                                               SEMIACTIVERADARGUIDEDMISSILECOUNT: self.myRuleHelper.getSemiActiveRadarGuidedMissileCount(),
                                               PASSIVERADARGUIDEDMISSILECOUNT: self.myRuleHelper.getPassiveRadarGuidedMissileCount()})
        except Exception as E:
            logger().savetxt(E, "InputErrorOcurredInMunition")

    def captureAssetTypeInfo(self):
        try:
            if ASSETTYPE in self.env.global_dict[self.env.processing_entity]:
                self.input_space[ASSETTYPE] = {}
                self.input_space.update({ASSETTYPE: self.env.global_dict[self.env.processing_entity][ASSETTYPE]})
            else:
                self.input_space[ASSETTYPE] = {}
                self.input_space[ASSETTYPE].update({ISPLATFORM: self.myRuleHelper.getIsPlatform(self.myGlobalID),
                                                    ISAIRPLATFORM: self.myRuleHelper.getIsAirPlatform(self.myGlobalID),
                                                    ISGROUNDPLATFORM: self.myRuleHelper.getIsGroundPlatform(self.myGlobalID),
                                                    ISAIRDEFENSEPLATFORM: self.myRuleHelper.getIsAirDefensePlatform(self.myGlobalID),
                                                    ISFIXEDWING: self.myRuleHelper.getIsFixedWing(self.myGlobalID),
                                                    ISROTARYWING: self.myRuleHelper.getIsRotaryWing(self.myGlobalID),
                                                    ISSHIP: self.myRuleHelper.getIsShip(self.myGlobalID),
                                                    ISSURFACEPLATFORM: self.myRuleHelper.getIsSurfacePlatform(self.myGlobalID),
                                                    ISLIFEFORM: self.myRuleHelper.getIsLifeform(self.myGlobalID),
                                                    ISGROUNDLIFEFORM: self.myRuleHelper.getIsGroundLifeform(self.myGlobalID),
                                                    ISCULTURALFEATURE: self.myRuleHelper.getIsCulturalFeature(self.myGlobalID),
                                                    ISUAV: self.myRuleHelper.getIsUnmannedAirPlatform(self.myGlobalID),
                                                    ISMUNITION: self.myRuleHelper.getIsMunition(self.myGlobalID)})
                self.env.global_dict[self.env.processing_entity][ASSETTYPE] = {}
                self.env.global_dict[self.env.processing_entity].update({ASSETTYPE: self.input_space[ASSETTYPE]})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInAssetType")

    def captureSensorInfo(self):
        try:
            self.input_space[SENSOR] = {}
            self.input_space[SENSOR].update({ISRWR: self.myRuleHelper.getIsReceivingRWRSignal(),
                                             ISMWR: self.myRuleHelper.getIsReceivingMWRSignal(),
                                             ISLWR: self.myRuleHelper.getIsReceivingLWRSignal(),
                                             TIMERWR: self.myRuleHelper.getRWRSystemCollisionTimeInSeconds(),
                                             TIMEMWR: self.myRuleHelper.getMWRSystemCollisionTimeInSeconds(),
                                             ANGLERWR: (self.myRuleHelper.getRWRSystemDetectedAngleInDegrees() - self.myRuleHelper.getMyHeading()) % 360,
                                             ANGLEMWR: (self.myRuleHelper.getMWRSystemDetectedAngleInDegrees() - self.myRuleHelper.getMyHeading()) % 360,
                                             ANGLELWR: (self.myRuleHelper.getLWRSystemDetectedAngleInDegrees() - self.myRuleHelper.getMyHeading()) % 360,
                                             ISRWRIO: self.myRuleHelper.getRWRMode(),
                                             ISMWRIO: self.myRuleHelper.getMWRMode(),
                                             ISLWRIO: self.myRuleHelper.getLWRMode(),
                                             ISRADARIO: self.myRuleHelper.isRadarInOperation(),
                                             ISRADARSEARCH: self.myRuleHelper.isRadarInSearchMode(),
                                             ISRADARTRACK: self.myRuleHelper.isRadarInTrackMode(),
                                             ISLASERIO: self.myRuleHelper.isLaserInOperation(),
                                             ISLASERDESIGNATION: self.myRuleHelper.isLaserInDesignationMode(),
                                             ISIFFIO: self.myRuleHelper.isIFFInOperation(),
                                             ISEOIO: self.myRuleHelper.isEODeviceInOperation(),
                                             ISIRJAMMERIO: self.myRuleHelper.isIRJammerInOperation(),
                                             ISLASERJAMMERIO: self.myRuleHelper.isLaserJammerInOperation(),
                                             ISRADARJAMMERIO: self.myRuleHelper.isRadarJammerInOperation(),
                                             ACQUISITION: self.getSensorAcquisitionInformation(),
                                             ISCHAFFINAUTO: self.myRuleHelper.isChaffInAutoMode(),
                                             ISFLAREINAUTO: self.myRuleHelper.isFlareInAutoMode(),
                                             ISSMOKEINAUTO: self.myRuleHelper.isSmokeInAutoMode()})
            if self.myRuleHelper.isRadarInTrackMode():
                self.input_space[SENSOR].update({TRACKEDENTITY: self.myRuleHelper.getTrackEntityGlobalId()})
                self.input_space[SENSOR].update({"tracked_entity_name": [self.myRuleHelper.getEntityName(self.myRuleHelper.getTrackEntityGlobalId())]})
            else:
                self.input_space[SENSOR].update({TRACKEDENTITY: 0})
                self.input_space[SENSOR].update({"tracked_entity_name": []})
            if self.myRuleHelper.isLaserInDesignationMode():
                self.input_space[SENSOR].update({DESIGNATEDENTITYID: self.myRuleHelper.getDesignatedEntityGlobalId()})
                self.input_space[SENSOR].update({DESIGNATEDENTITYNAME: [self.myRuleHelper.getEntityName(self.myRuleHelper.getDesignatedEntityGlobalId())]})
            else:
                self.input_space[SENSOR].update({DESIGNATEDENTITYID: 0})
                self.input_space[SENSOR].update({DESIGNATEDENTITYNAME: []})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInSensor")

    ################################################ HAS PO ENTITY & PO ENTITY LIST ############################################
    def captureHasPOEntityInfo(self):
        try:
            if HASPOENTITY in self.env.global_dict[self.env.processing_entity]:
                self.input_space[HASPOENTITY] = {}
                self.input_space.update({HASPOENTITY: self.env.global_dict[self.env.processing_entity][HASPOENTITY]})
            else:
                self.input_space[HASPOENTITY] = {}
                self.input_space.update({HASPOENTITY: self.myRuleHelper.hasPOEntity()})
                self.env.global_dict[self.env.processing_entity][HASPOENTITY] = {}
                self.env.global_dict[self.env.processing_entity].update({HASPOENTITY: self.input_space[HASPOENTITY]})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInHasPOEntity")

    def capturePOEntityInfo(self):
        try:
            self.input_space[POENTITY] = {}
            self.input_space.update({POENTITY: self.getPOEntityInformation()})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInPOEntity")

    def getPOEntityInformation(self):
        POEntityList = []
        try:
            if self.myRuleHelper.hasPOEntity():
                POEntityList.append(self.myRuleHelper.getEntityName(self.myRuleHelper.getPOEntityGlobalId()))
        except Exception as E:
            print("PO Entity collector")
            print(E)
        return POEntityList

    ###################################################### HAS SUBSYSTEM ######################################################
    def captureHasSubsystemInfo(self):
        try:

            if HASSUBSYSTEM in self.env.global_dict[self.env.processing_entity]:
                self.input_space[HASSUBSYSTEM] = {}
                self.input_space.update({HASSUBSYSTEM: self.env.global_dict[self.env.processing_entity][HASSUBSYSTEM]})
            else:
                self.input_space[HASSUBSYSTEM] = {}
                self.input_space[HASSUBSYSTEM].update({HASRADARSUBSYSTEM: self.myRuleHelper.hasRadarSubsystem(),
                                                       HASLASERSUBSYSTEM: self.myRuleHelper.hasLaserSubsystem(),
                                                       HASIFFSUBSYSTEM: self.myRuleHelper.hasIFFSubsystem(),
                                                       HASEOSUBSYSTEM: self.myRuleHelper.hasEOSubsystem(),
                                                       HASRWRSUBSYSTEM: self.myRuleHelper.hasRWRSubsystem(),
                                                       HASMWRSUBSYSTEM: self.myRuleHelper.hasMWRSubsystem(),
                                                       HASLWRSUBSYSTEM: self.myRuleHelper.hasLWRSubsystem(),
                                                       HASRADARJAMMERSUBSYSTEM: self.myRuleHelper.hasRadarJammerSubsystem(),
                                                       HASLASERJAMMERSUBSYSTEM: self.myRuleHelper.hasLaserJammerSubsystem(),
                                                       HASIRJAMMERSUBSYSTEM: self.myRuleHelper.hasIRJammerSubsystem()})
                self.env.global_dict[self.env.processing_entity][HASSUBSYSTEM] = {}
                self.env.global_dict[self.env.processing_entity].update({HASSUBSYSTEM: self.input_space[HASSUBSYSTEM]})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInHasSubsystem")

    ###################################################### ENTITY IN RANGE ##################################################################
    def captureEntityInRangeInfo(self):
        try:
            self.input_space[ENTITYINRANGE] = {}
            firing, dive, gun = self.getEntityInRangeInfo()
            laser = self.getEntityInLaserRangeInfo()
            self.input_space[ENTITYINRANGE].update({DIVERANGE: dive,
                                                    FIRINGRANGE: firing,
                                                    ENTITYINGUNRANGE: gun,
                                                    ENTITYINLASERRANGE: laser})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredForEntityInRangeInfo")

    def getEntityInLaserRangeInfo(self):
        entities_in_laser_range = []
        self._resetEntityList()
        self.myRuleHelper.eliminateWRTAcquisition(self.entityInstanceListObject, True)
        size = self.entityInstanceListObject.getEntityInstanceSize()
        for i in range(0, size):
            entity_id = self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)
            if self.myRuleHelper.isEntityInLaserRange(entity_id):
                entities_in_laser_range.append(self.myRuleHelper.getEntityName(entity_id))
        return entities_in_laser_range


    def getEntityInRangeInfo(self):
        none_list, laser_list, ar_list, ir_list, sar_list, pr_list, eo_list = [],[],[],[],[],[],[]
        none_dive_list, laser_dive_list, ar_dive_list, ir_dive_list, sar_dive_list, pr_dive_list, eo_dive_list = [],[],[],[],[],[],[]
        gun_list = []
        entity_distances = self.input_space[MOTION][DISTANCE]
        for entity, distance in entity_distances.items():
            if distance <= self.none_range * self.env.standardFiringRangeCoeff:
                none_list.append(entity)
            if distance <= self.laser_range * self.env.standardFiringRangeCoeff:
                laser_list.append(entity)
            if distance <= self.ar_range * self.env.activeRadarFiringRangeCoeff:
                ar_list.append(entity)
            if distance <= self.ir_range * self.env.IRFiringRangeCoeff:
                ir_list.append(entity)
            if distance <= self.sar_range * self.env.standardFiringRangeCoeff:
                sar_list.append(entity)
            if distance <= self.pr_range * self.env.standardFiringRangeCoeff:
                pr_list.append(entity)
            if distance <= self.eo_range * self.env.EOFiringrangeCoeff:
                eo_list.append(entity)

            if distance <= self.none_range + self.env.diveRangeCoeff:
                none_dive_list.append(entity)
            if distance <= self.laser_range + self.env.diveRangeCoeff:
                laser_dive_list.append(entity)
            if distance <= self.ar_range + self.env.diveRangeCoeff:
                ar_dive_list.append(entity)
            if distance <= self.ir_range + self.env.diveRangeCoeff:
                ir_dive_list.append(entity)
            if distance <= self.sar_range + self.env.diveRangeCoeff:
                sar_dive_list.append(entity)
            if distance <= self.pr_range + self.env.diveRangeCoeff:
                pr_dive_list.append(entity)
            if distance <= self.eo_range + self.env.diveRangeCoeff:
                eo_dive_list.append(entity)

            if distance <= self.gun_range:
                gun_list.append(entity)

        firing_dict = {}
        firing_dict.update({NONEMISSILE: none_list,
                            LASERGUIDEDMISSILE: laser_list,
                            ACTIVERADARGUIDEDMISSILE: ar_list,
                            IRGUIDEDMISSILE: ir_list,
                            SEMIACTIVERADARGUIDEDMISSILE: sar_list,
                            PASSIVERADARGUIDEDMISSILE: pr_list,
                            EOGUIDEDMISSILE: eo_list
                            })
        dive_dict = {}
        dive_dict.update({NONEMISSILE: none_dive_list,
                            LASERGUIDEDMISSILE: laser_dive_list,
                            ACTIVERADARGUIDEDMISSILE: ar_dive_list,
                            IRGUIDEDMISSILE: ir_dive_list,
                            SEMIACTIVERADARGUIDEDMISSILE: sar_dive_list,
                            PASSIVERADARGUIDEDMISSILE: pr_dive_list,
                            EOGUIDEDMISSILE: eo_dive_list
                            })
        return firing_dict, dive_dict, gun_list


################################################### COMMUNICATION ############################################################
    def captureCommInfo(self):
        try:
            self.input_space[COMM] = {}
            targets, commanders = self.getTargetAndCommanderInfo()
            # print(commanders)
            self.input_space[COMM].update({ENGAGECOMMAND: self.env.engage_command,
                                           TARGETENTITY: targets,
                                           COMMANDERENTITY: commanders,
                                           CLOSERADARCOMMAND: self.env.closeradar_command,
                                           TURNCOMMAND: self.env.turn_command,
                                           TARGETMISSILE: self.getTargetMissileInformation(),
                                           FUELCOMMAND: self.env.fuel_command})
            if self.myGlobalID == self.env.myEntityInstanceGlobalId:
                self.input_space[COMM].update({MYMESSAGEBOX: list(self.my_message_history)})
            else:
                self.input_space[COMM].update({MYMESSAGEBOX: list(deque([None]*10, maxlen=10))})

            my_name = self.myRuleHelper.getEntityName(self.myGlobalID)
            if my_name not in self.fired_history:
                self.fired_history.update({my_name: deque([False]*10, maxlen=10)})
        except Exception as E:
            logger().savetxt(E,"BecomingInputErrorForCaptureCommInfo")

    def getTargetAndCommanderInfo(self):
        targetEntityList = []
        commanderEntityList = []
        for i, ID in enumerate(self.env.Globalidlist):
            if ID == self.env.myEnemyGlobalID:
                targetEntityList.append(self.env.entitynamelist[i])
            if ID == self.env.commanderID:
                commanderEntityList.append(self.env.entitynamelist[i])
        return targetEntityList, commanderEntityList

    def getTargetMissileInformation(self):
        target_missile_list = []
        self._resetEntityList()
        self.myRuleHelper.getAllMissileEntityInstances(self.entityInstanceListObject)
        self.myRuleHelper.eliminateWRTDistance(self.entityInstanceListObject, 3, 200000)
        size = self.entityInstanceListObject.getEntityInstanceSize()
        for i in range(0, size):
            if self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i) == self.env.myEnemyMissileGlobalID:
                target_missile_list.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
        return target_missile_list

########################################################### ROUTE #################################################
    def captureRouteInfo(self):
        try:
            coordinate = self.myRuleHelper.getMyRoutePointCoordinate(self.myRuleHelper.getMyLastRoutePoint())
            self.input_space[ROUTE] = {}
            self.input_space[ROUTE].update({HASROUTE: self.myRuleHelper.hasRoute(),
                                            ISROUTEACTIVE: self.myRuleHelper.getIsRouteActive(),
                                            ROUTEPOINTCOORDINATE: [coordinate.latitude, coordinate.longitude, coordinate.altitude]
                                            })
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInCaptureRouteInfo_inputcollector")


    def captureIFFInfo(self):
        try:
            iff_dict, force_dict = self.getIFFAndForceInfo()
            self.input_space.update({IFF: iff_dict,
                                     FORCE: force_dict})
        except Exception as E:
            logger().savetxt(E,"BecomingInputErrorForIFFandForce")

    def captureRelatives(self):
        RelativeBearingAngleDict = {}
        BearingAngleDict = {}
        DistanceDict = {}
        for i, ID in enumerate(self.env.Globalidlist):
            if ID == self.myGlobalID:
                continue
            name = str(self.env.entitynamelist[i])
            RelativeBearingAngleDict.update({name: self.myRuleHelper.getRelativeBearingWRTEntity(ID)})
            BearingAngleDict.update({name: self.myRuleHelper.getBearingWRTEntity(ID)})
            DistanceDict.update({name: self.myRuleHelper.getDistanceWRTEntity(ID)})
        return RelativeBearingAngleDict, BearingAngleDict, DistanceDict

    def getIFFAndForceInfo(self):
        IFFDict={}
        ForceDict = {}
        dict_keys = [(IFF_UNKNOWN,FORCE_ENEMY), (IFF_FRIEND,FORCE_FRIEND), (IFF_HOSTILE,None)]
        for k, dict_key in enumerate(dict_keys):
            self._resetEntityList()
            self.myRuleHelper.eliminateWRTIFFInterrogationResult(self.entityInstanceListObject, k)
            size = self.entityInstanceListObject.getEntityInstanceSize()
            IFFlist = []
            for i in range(0, size):
                IFFlist.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
            IFFDict.update({dict_key[0]: IFFlist})
            if k < 2:
                self._resetEntityList()
                self.myRuleHelper.eliminateWRTForce(self.entityInstanceListObject, k + 1)
                size = self.entityInstanceListObject.getEntityInstanceSize()
                Forcelist = []
                for i in range(0, size):
                    Forcelist.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
                ForceDict.update({dict_key[1]: Forcelist})
        return IFFDict, ForceDict

    def getSensorAcquisitionInformation(self):
        SensorList = []
        self._resetEntityList()
        self.myRuleHelper.eliminateWRTAcquisition(self.entityInstanceListObject, True)
        size = self.entityInstanceListObject.getEntityInstanceSize()
        for i in range(0, size):
            SensorList.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
        return SensorList

    def captureDISCodeInfo(self):
        try:
            if DISCODE in self.env.global_dict[self.env.processing_entity]:
                self.input_space[DISCODE] = {}
                self.input_space.update({DISCODE: self.env.global_dict[self.env.processing_entity][DISCODE]})
            else:
                self.input_space[DISCODE] = {}
                self.input_space[DISCODE].update({ENTITYKIND: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).EntityKind,
                                                  COUNTRYCODE: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).CountryCode,
                                                  CATEGORY: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).Category,
                                                  SUBCATEGORY: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).Subcategory,
                                                  SPECIFIC: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).Specific,
                                                  EXTRA: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).Extra,
                                                  DOMAIN: self.myRuleHelper.getSpecifiedEntityDISCode(self.myGlobalID).Domain})
                self.env.global_dict[self.env.processing_entity][DISCODE] = {}
                self.env.global_dict[self.env.processing_entity].update({DISCODE: self.input_space[DISCODE]})
        except Exception as E:
            logger().savetxt(E, "InputErrorOccuredInCaptureDISCodeInfo_inputcollector")

    def captureWheatherConditionsInfo(self):
        try:
            self.input_space[WEATHERCONDITIONS] = {}
            self.input_space[WEATHERCONDITIONS].update({PRECIPITATIONTYPE: self.myRuleHelper.getPrecipitationType(),
                                                        PRECIPITATIONLEVEL: self.myRuleHelper.getPrecipitationLevel(),
                                                        CLOUDCOVERAGELEVEL: self.myRuleHelper.getInCloudCoverageLevel(),
                                                        CLOUDTOPINMETERS: self.myRuleHelper. getCloudTopInMeters(),
                                                        CLOUDBOTTOMINMETERS: self.myRuleHelper.getCloudBottomInMeters(),
                                                        VISIBILITYINMETERS: self.myRuleHelper.getVisibilityInMeters(),
                                                        TEMPERATUREINCELCIUS: self.myRuleHelper.getTemperatureInCelcius(),
                                                        WINDDIRECTIONINDEGREES: self.myRuleHelper.getWindDirectionInDegrees(),
                                                        WINDSPEEDINKNOTS: self.myRuleHelper.getWindSpeedInKnots(),
                                                        MOONILLUMINATION: self.myRuleHelper.getMoonIlluminationPercentage()})
        except Exception as E:
            logger().savetxt(E, "inputcollector_wheather_error")

    def captureTimeInfo(self):
        try:
            self.input_space[TIME] = {}
            self.input_space[TIME].update({CURRENTSCENARIODATETIMEYEAR: self.myRuleHelper.getCurrentScenarioDateTime().year,
                                           CURRENTSCENARIODATETIMEMONTH: self.myRuleHelper.getCurrentScenarioDateTime().month,
                                           CURRENTSCENARIODATETIMEDAY: self.myRuleHelper.getCurrentScenarioDateTime().day,
                                           CURRENTSCENARIODATETIMEHOUR: self.myRuleHelper.getCurrentScenarioDateTime().hour,
                                           CURRENTSCENARIODATETIMEMINUTE: self.myRuleHelper.getCurrentScenarioDateTime().minute,
                                           CURRENTSCENARIODATETIMESECOND: self.myRuleHelper.getCurrentScenarioDateTime().second,
                                           SECTIONOFDAY: self.myRuleHelper.getSectionOfDay()})
        except Exception as E:
            logger().savetxt(E, "inputcollector_time_error")

    def captureFuelInfo(self):
        try:
            self.input_space[FUEL] = {}
            self.input_space[FUEL].update({FUELAMOUNT: self.myRuleHelper.getMyFuelAmountInPercentage()})
        except Exception as E:
            logger().savetxt(E, "inputcollector_fuel_error")

    def captureFormationInfo(self):
        try:
            self.input_space[FORMATION] = {}
            self.input_space[FORMATION].update({FORMATIONORDER: self.myRuleHelper.getFormationOrder()})
        except Exception as E:
            logger().savetxt(E, "inputcollector_formation_error")

    def captureMissileThreatInfo(self):
        try:
            self.env.ARMDetected = False
            dummy_entity_instance_list_object = self.entityInstanceListObject
            self.myRuleHelper.getAllMissileEntityInstances(dummy_entity_instance_list_object)
            self.myRuleHelper.eliminateWRTDistance(dummy_entity_instance_list_object, 3, 200000)
            self.myRuleHelper.eliminateWRTAcquisition(dummy_entity_instance_list_object, True)
            size = dummy_entity_instance_list_object.getEntityInstanceSize()
            if size > 0:
                for i in range(0, size):
                    potential_missile_id = dummy_entity_instance_list_object.getEntityInstanceGlobalIdAtIndex(i)
                    potential_missile_DIS = self.myRuleHelper.getSpecifiedEntityDISCode(potential_missile_id)
                    if (potential_missile_DIS.Domain == 4):
                        self.env.ARMDetected = True
                        self.env.closeRadarTimerIsStarted = True
                        break

            self.input_space[MISSILETHREAT] = {}
            self.input_space[MISSILETHREAT].update({ARMTHREAT: self.env.ARMDetected})

        except Exception as E:
            logger().savetxt(E, "inputcollector_ARMDetected_error")

    def captureThreatsInfo(self):
        try:
            self.input_space[THREATS] = {}

            fire_threats_list = []
            self._resetEntityList()
            self.myRuleHelper.eliminateWRTAcquisition(self.entityInstanceListObject, True)
            self.myRuleHelper.eliminateWRTFireThreats(self.entityInstanceListObject, True)
            size = self.entityInstanceListObject.getEntityInstanceSize()
            for i in range(0, size):
                fire_threats_list.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
            self.input_space[THREATS].update({FIRETHREATS: fire_threats_list})

            radar_threats_list = []
            self._resetEntityList()
            self.myRuleHelper.eliminateWRTAcquisition(self.entityInstanceListObject, True)
            self.myRuleHelper.eliminateWRTRadarThreats(self.entityInstanceListObject, True)
            size = self.entityInstanceListObject.getEntityInstanceSize()
            for i in range(0, size):
                radar_threats_list.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
            self.input_space[THREATS].update({RADARTHREATS: radar_threats_list})

            laser_threats_list = []
            self._resetEntityList()
            self.myRuleHelper.eliminateWRTAcquisition(self.entityInstanceListObject, True)
            self.myRuleHelper.eliminateWRTLaserThreats(self.entityInstanceListObject, True)
            size = self.entityInstanceListObject.getEntityInstanceSize()
            for i in range(0, size):
                laser_threats_list.append(self.myRuleHelper.getEntityName(self.entityInstanceListObject.getEntityInstanceGlobalIdAtIndex(i)))
            self.input_space[THREATS].update({LASERTHREATS: laser_threats_list})

        except Exception as E:
            logger().savetxt(E, "inputcollector_threats_error")


    def makestate(self, globalid):

        self.myGlobalID = globalid
        self.myRuleHelper = hvlruleutil.RuleHelper(globalid)
        self.input_space = {}
        self.captureGeneralInfo()
        self.captureMotionInfo()
        self.captureIFFInfo()
        self.captureMunitionInfo()
        self.captureAssetTypeInfo()
        self.captureSensorInfo()
        self.captureHasSubsystemInfo()
        self.captureHasPOEntityInfo()
        self.capturePOEntityInfo()
        self.captureEntityInRangeInfo()
        self.captureRouteInfo()
        self.captureCommInfo()
        self.captureDISCodeInfo()
        self.captureWheatherConditionsInfo()
        self.captureTimeInfo()
        self.captureFuelInfo()
        self.captureFormationInfo()
        self.captureMissileThreatInfo()
        self.captureThreatsInfo()

        return self.input_space





