(window.__LOADABLE_LOADED_CHUNKS__=window.__LOADABLE_LOADED_CHUNKS__||[]).push([[2],{"1iNE":function(n,e,t){var r=t("A90E"),o=t("QqLw"),i=t("MMmD"),u=t("4qC0"),_=t("Z1HP");n.exports=function(n){if(null==n)return 0;if(i(n))return u(n)?_(n):n.length;var e=o(n);return"[object Map]"==e||"[object Set]"==e?n.size:r(n).length}},EwQA:function(n,e,t){var r=t("zZ0H");n.exports=function(n){return"function"==typeof n?n:r}},"L5/0":function(n,e,t){var r=t("dt0z");n.exports=function(n){return r(n).toLowerCase()}},OFL0:function(n,e,t){var r=t("lvO4"),o=t("4sDh");n.exports=function(n,e){return null!=n&&o(n,e,r)}},Xdxp:function(n,e,t){var r=t("g4R6"),o=t("zoYe"),i=t("Sxd8"),u=t("dt0z");n.exports=function(n,e,t){return n=u(n),t=null==t?0:r(i(t),0,n.length),e=o(e),n.slice(t,t+e.length)==e}},Z1HP:function(n,e,t){var r=t("ycre"),o=t("quyA"),i=t("q4HE");n.exports=function(n){return o(n)?i(n):r(n)}},bNQv:function(n,e,t){var r=t("gFfm"),o=t("SKAX"),i=t("EwQA"),u=t("Z0cm");n.exports=function(n,e){return(u(n)?r:o)(n,i(e))}},g4R6:function(n,e){n.exports=function(n,e,t){return n==n&&(void 0!==t&&(n=n<=t?n:t),void 0!==e&&(n=n>=e?n:e)),n}},lvO4:function(n,e){var t=Object.prototype.hasOwnProperty;n.exports=function(n,e){return null!=n&&t.call(n,e)}},q4HE:function(n,e){var t="[\\ud800-\\udfff]",r="[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]",o="\\ud83c[\\udffb-\\udfff]",i="[^\\ud800-\\udfff]",u="(?:\\ud83c[\\udde6-\\uddff]){2}",_="[\\ud800-\\udbff][\\udc00-\\udfff]",E="(?:"+r+"|"+o+")"+"?",l="[\\ufe0e\\ufe0f]?"+E+("(?:\\u200d(?:"+[i,u,_].join("|")+")[\\ufe0e\\ufe0f]?"+E+")*"),O="(?:"+[i+r+"?",r,u,_,t].join("|")+")",c=RegExp(o+"(?="+o+")|"+O+l,"g");n.exports=function(n){for(var e=c.lastIndex=0;c.test(n);)++e;return e}},sUTr:function(n,e,t){"use strict";(function(n){t.d(e,"a",(function(){return o}));var r,o,i,u,_=t("mwIZ"),E=t.n(_),l=t("Xdxp"),O=t.n(l),c=t("L5/0"),a=t.n(c);t("bNQv"),t("OFL0"),t("1iNE");!function(n){n.CLOUD="cloud",n.CPD="cpd"}(r||(r={})),function(n){n.AVAILABLE="available",n.CONSTRICTED="constricted",n.DEPRECATED="deprecated",n.INVALID="invalid",n.REMOVED="removed"}(o||(o={})),function(n){n.CONSTRICTED_PYTHON_38_ON_CLOUD="python38IsConstrictedOnCloud",n.CONSTRICTED_SPARK_PYTHON_38="sparkPython38IsConstriced",n.CONSTRICTED_NLP_BETA="nlpBetaConstricted",n.CONSTRICTED_NOTEBOOK_GPU_K80="notebookGpuK80IsConstricted",n.REMOVED_NLP_BETA="nlpBetaRemoved",n.DEPRECATED_R36_PYTHON_38="r36Python38IsDeprected",n.DEPRECATED_R36_PYTHON_39="r36Python39IsDeprected",n.REMOVED_NOTEBOOK_R36_ON_CLOUD="notebookR36IsRemovedOnCloud",n.REMOVED_RSTUDIO_R36="rstudioR36IsRemoved",n.DEPRECATED_NOTEBOOK_GPU_K80="notebookGpuK80IsDeprecated",n.REMOVED_NOTEBOOK_PYTHON_39="notebookPy39IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_39_ON_CLOUD="notebookPy39IsRemovedOnCloud",n.DEPRECATED_SPARK_R36_ON_CLOUD="sparkR36IsDeprecatedOnCloud",n.REMOVED_SPARK_R36_ON_CLOUD="sparkR36IsRemovedOnCloud",n.REMOVED_SPARK_R36="sparkR36IsRemoved",n.DEPRECATED_SPARK_PYTHON_39="sparkPy39IsDeprecated",n.INVALID_DO_ON_FREE_HARDWARE="notebookDecisionOptimizationOnFreeHardwareIsInvalid",n.INVALID_ENVIRONMENT="invalidEnvironment",n.INVALID_FOR_WATONSX_CHALLENGE="invalidEnvironmentForWatsonXChallenge",n.REMOVED_SATELLITE_NOTEBOOK="satelliteNotebookIsRemoved",n.REMOVED_COMPUTE_FROM_PLAN="computeIsNotAllowed",n.REMOVED_FREE_HARDWARE="freeHardwareIsRemoved",n.REMOVED_HARDWARE_CONFIG_FROM_PLAN="hardwareSizeIsNotAllowed",n.REMOVED_IAE_FROM_PLAN="IAEEnvironmentIsNotAllowed",n.REMOVED_GPU_NON_CE="notebookNonCEGpuIsRemoved",n.REMOVED_PREMIUM_PYTHON_NON_CE="notebookNonCEPremiumPythonIsRemoved",n.REMOVED_PYTHON_NON_CE="notebookNonCEPythonIsRemoved",n.REMOVED_NOTEBOOK_BETA_GPU_V100="notebookBetaGpuV100IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_27="notebookPython27IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_35="notebookPython35IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_35_DO="notebookDecisionOptimizationPython35IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_36="notebookPython36IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_36_DO="notebookDecisionOptimizationPython36IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_37="notebookPython37IsRemoved",n.REMOVED_NOTEBOOK_PYTHON_38="notebookPython38IsRemoved",n.REMOVED_PYTHON_38_ON_CLOUD="python38IsRemovedOnCloud",n.REMOVED_NOTEBOOK_R34="notebookR34IsRemoved",n.REMOVED_NOTEBOOK_R36="notebookR36IsRemoved",n.REMOVED_NOTEBOOK_R36_ON_PYTHON38="notebookR36Python38IsRemoved",n.REMOVED_NOTEBOOK_R36_ON_PYTHON37="notebookR36OnPython37IsRemoved",n.REMOVED_SPARK_23="spark23IsRemoved",n.REMOVED_SPARK_24="spark24IsRemoved",n.REMOVED_SPARK_30="spark30IsRemoved",n.REMOVED_SPARK_32="spark32IsRemoved",n.REMOVED_SPARK_23_ON_CLOUD="spark23IsRemovedOnCloud",n.REMOVED_SPARK_24_ON_CLOUD="spark24IsRemovedOnCloud",n.REMOVED_SPARK_30_ON_CLOUD="spark30IsRemovedOnCloud",n.REMOVED_SPARK_32_ON_CLOUD="spark32IsRemovedOnCloud",n.REMOVED_NOTEBOOK_GPU_K80="notebookGpuK80IsRemoved",n.DEPRECATED_SCALA="scalaIsDeprecated",n.DEPRECATED_SPARK_30="spark30IsDeprecated",n.DEPRECATED_SPARK_32="spark32IsDeprecated",n.DEPRECATED_SPARK_30_ON_CLOUD="spark30IsDeprecatedOnCloud",n.REMOVED_SCALA="scalaIsRemoved",n.CONSTRICTED_SCALA="scalaIsConstricted",n.CONSTRICTED_SPARK_30_ON_CLOUD="spark30IsConstrictedOnCloud",n.CONSTRICTED_SPARK_32_ON_CLOUD="spark32IsConstrictedOnCloud",n.DEPRECATED_SPARK_32_ON_CLOUD="spark32IsDeprecatedOnCloud",n.REMOVED_SPARK_AS_A_SERVICE="sparkAsAServiceIsRemoved",n.REMOVED_SPARK_PYTHON_27="sparkPython27IsRemoved",n.REMOVED_SPARK_PYTHON_35="sparkPython35IsRemoved",n.REMOVED_SPARK_PYTHON_36="sparkPython36IsRemoved",n.REMOVED_SPARK_PYTHON_37="sparkPython37IsRemoved",n.REMOVED_SPARK_PYTHON_38="sparkPython38IsRemoved",n.REMOVED_SPARK_PYTHON_39="sparkPython39IsRemoved",n.MISSING_RUNTIME_DEF_ID_CPD="missingRuntimeDefinitionIDOnCPD",n.DEPRECATED_ENV_SPEC="deprecatedEnvSpec",n.DEPRECATED_SW_SPEC="deprecatedSwSpec",n.DEPRECATED_HW_SPEC="deprecatedHwSpec",n.CONSTRICTED_ENV_SPEC="constrictedEnvSpec",n.CONSTRICTED_SW_SPEC="constrictedSwSpec",n.CONSTRICTED_HW_SPEC="constrictedHwSpec",n.RETIRED_ENV_SPEC="retiredEnvSpec",n.RETIRED_SW_SPEC="retiredSwSpec",n.RETIRED_HW_SPEC="retiredHwSpec"}(i||(i={})),function(n){n.JUPYTER_PY35="6d6ce89b-776d-4b5f-989c-934d2a8e5cf8",n.JUPYTER_PY35_GPU="e580c309-0cb0-4cec-bee0-4faf2679312d",n.JUPYTER_PY36="7026e5ef-2367-4f39-8bab-37b953245563",n.JUPYTER_PY37="f61bafba-b082-4d8a-ad6f-bbe28e4790ac",n.JUPYTER_PY37_LEGACY="91d7e404-5236-4187-b976-7e896f70b178",n.JUPYTER_PY38="50cf5cb3-bb57-45b5-9f8c-e68671180072",n.JUPYTER_PY39="204dac19-7d94-4692-a4c8-2acd77e28792",n.JUPYTER_PY39_GPU="0fb69fde-6805-5152-9238-d488ee7f4dc1",n.JUPYTER_R="d9212498-8e11-45a7-b4bc-308a3b8e2cd7",n.JUPYTER_R36="628d508f-6e4b-4ae9-8c9a-62aa1f179a27",n.JUPYTER_R36_PY39="98c9bba2-724d-552e-8395-be3d44bfbf51",n.JUPYTER_LAB="fd807a24-0966-4b6e-95a8-7088e5a6abdb",n.JUYPTER_LAB_PY37="82df8758-c5f5-490e-a2a5-51457759f18d",n.LAB_PY37_LEGACY="ae43ae62-ad4d-4d65-8ada-2680108146e8",n.JUPYTER_LAB_PY38="dcc728be-d615-4cff-a950-aebe9f06df2f",n.JUPYTER_LAB_PY39="7a129591-7702-4bdc-a570-47859d38b9d5",n.JUPYTER_LAB_PY39_GPU="17f0181e-1840-5e04-bdba-237d53266c0c",n.RSTUDIO="45b69e8f-e2af-45da-ad5f-244249faa843",n.RSTUDIO_R36="6cc6b9d2-87e8-5355-bde5-aa70b1b50c3c",n.RSTUDIO_231_R42="7fbf1305-59a7-5b77-b881-1416ce2ee903",n.RSTUDIO_R42="42c36a39-fcc1-5117-8ff6-1d4523e0d6a6",n.RUNTIME_231_PY310_CUDA="f31eb7d1-a784-5a74-a40b-579989aacfcb",n.RUNTIME_231_PY310_XC="b1a3e58f-ff58-59d7-8dc5-62042e42a538",n.RUNTIME_231_PY310="336b29df-e0e1-5e7d-b6a5-f6ab722625b2",n.RUNTIME_231_R42="a1275ec7-2602-52bd-a193-44ba137265df",n.RUNTIME_232_PY311_CUDA="03a1d054-8b90-5e3f-9c7a-4448ce3fcf68",n.RUNTIME_232_PY311_XC="c16ef4e9-b4cf-5e04-bdd9-0e00e17f344b",n.RUNTIME_232_PY311="090a9075-d969-56c2-a8f3-fb94a68f3d84",n.RUNTIME_232_R42="c7226fc1-a1a5-5bc0-9946-3477d9f9b93f",n.RUNTIME221_PY39_CUDA="26215f05-08c3-5a41-a1b0-da66306ce658",n.RUNTIME221_PY39_DO="a7e7dbf1-1d03-5544-994d-e5ec845ce99a",n.RUNTIME221_PY39="12b83a17-24d8-5082-900f-0ab31fbfd3cb",n.RUNTIME221_R36="018ebea5-1d1f-5fec-b93e-5e2ab30e7f38",n.RUNTIME222_PY310_CUDA="8ef391e4-ef58-5d46-b078-a82c211c1058",n.RUNTIME222_PY310_XC="5e8cddff-db4a-5a6a-b8aa-2d4af9864dab",n.RUNTIME222_PY310="b56101f1-309d-549b-a849-eaa63f77b2fb",n.RUNTIME222_R42="ec0a3d28-08f7-556c-9674-ca7c2dba30bd",n.RUNTIME_221_PY39_NNPA="6d93276f-1abb-523f-bf07-7c05deb7b0f9"}(u||(u={}));function R(n){return"referenced"===E()(n,"entity.environment.spec_type","")}function f(n){return R(n)}function s(n,e=null){return R(n)?E()(n,"entity.environment.hardware_specification.entity.hardware_specification",e):E()(n,"entity.environment.hardware_specification")}function d(n,e=null){return R(n)?E()(n,"entity.environment.software_specification.entity.software_specification",e):E()(n,"entity.environment.software_specification")}function m(n,e=null){return E()(function(n,e=null){return R(n)?E()(n,"entity.environment.tools_specification",e):d(n)}(n),"supported_kernels",e)}function v(n,e=null){const t=R(n)?"nodes.cpu.units":"num_cpu";return E()(s(n),t,e)}function D(n,e=null){const t=R(n)?"spark.driver.cpu.units":"spark.driver.cores";return E()(s(n),t,e)}function P(n,e=null){const t=R(n)?"spark.num_executors":"spark.executor.instances";return E()(s(n),t,e)}function p(n,e=null){const t=R(n)?"spark.executor.cpu.units":"spark.executor.cores";return E()(s(n),t,e)}function N(n,e=null){const t=R(n)?"nodes.gpu.name":"gpu.subModel";return E()(s(n),t,e)}function T(n,e){const t=e.compute;return function(n){const e=n.compute,t=n.entitlements,r=E()(e,"type"),o=E()(t,"properties.services."+r+".enabled");return e&&!o}({compute:t,entitlements:e.entitlements})?k(t,null,o.REMOVED,i.REMOVED_COMPUTE_FROM_PLAN):null}function b(n){let e;return n.environment?e=function(n,e="notebook"){return E()(n,"entity.environment.sub_type",e)}(n.environment):n.compute&&(e=E()(n.compute,"type")),e===n.type}function M({environment:n}){return E()(n,"entity.environment.runtime_definition")}function V(n){const e=n.environment,t=n.language,r=m(e);let o;return Array.isArray(r)&&r.length&&(o=function(n,e=null){return E()(n,"language",e)}(r[0])),O()(a()(o),t)}function C(n){const e=n.environment,t=n.version,r=m(e);let o;return Array.isArray(r)&&r.length&&(o=function(n,e=null){return E()(n,"version",e)}(r[0])),a()(o)===t}function A(n){const e=n.environment,t=n.environmentPackage,r=n.environmentPackageRegExp,o=function(n,e=null){return R(n)?e:E()(n,"entity.environment.software_specification.package",e)}(e);return r?r.test(o):t===o}function I(n){const e=n.environment,t=n.version,r=function(n,e=null){const t=R(n)?"software_configuration.platform.version":"spark_version";return E()(d(n),t,e)}(e);return a()(r)===t}function y(n){return function(n){return E()(s(n),"free",!1)}(n.environment)}function S(n){const e=n.environment,t=n.subModel;return N(e)===t}function Y(n){const e=n.environment,t=n.limit,r=Number(p(e));return r&&t&&r>t}function K({environment:n,softwareSpecId:e}){return function(n,e=null){return R(n)?E()(n,"entity.environment.software_specification.entity.software_specification.base_software_specification.guid",e):e}(n,function(n,e=null){return R(n)?E()(n,"entity.environment.software_specification.metadata.asset_id",e):e}(n))===e}function k(n,e,t,r){const o=E()(n,"metadata.asset_id"),i=E()(e,"guid");return{guid:o||i,status:t,reasonCode:r}}function U(n){const e=n.environment;if(!e)return null;for(const n of["retired","constricted","deprecated"])for(const t of["entity.environment.software_specification.metadata.life_cycle","entity.environment.hardware_specification.metadata.life_cycle","metadata.life_cycle"]){const r=t+"."+n;if(E()(e,r,!1))return H(e,t,n)}return null}function H(n,e,t){let r,u;switch(t){case"deprecated":switch(o.DEPRECATED,e){case"metadata.life_cycle":i.DEPRECATED_ENV_SPEC;break;case"entity.environment.software_specification.metadata.life_cycle":i.DEPRECATED_SW_SPEC;break;case"entity.environment.hardware_specification.metadata.life_cycle":i.DEPRECATED_HW_SPEC}break;case"constricted":switch(e){case"metadata.life_cycle":i.CONSTRICTED_ENV_SPEC;break;case"entity.environment.software_specification.metadata.life_cycle":i.CONSTRICTED_SW_SPEC;break;case"entity.environment.hardware_specification.metadata.life_cycle":i.CONSTRICTED_HW_SPEC}o.CONSTRICTED;break;case"retired":switch(o.REMOVED,e){case"metadata.life_cycle":i.RETIRED_ENV_SPEC;break;case"entity.environment.software_specification.metadata.life_cycle":i.RETIRED_SW_SPEC;break;case"entity.environment.hardware_specification.metadata.life_cycle":i.RETIRED_HW_SPEC}}return k(n,null,r,u)}(function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"scala"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SCALA)}).bind(null,{errorCode:i.REMOVED_SCALA}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"3.2"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_32_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_SPARK_32_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackageRegExp:/-premium$/}))return null;if(!y({environment:t}))return null;return k(t,null,o.INVALID,i.INVALID_DO_ON_FREE_HARDWARE)}.bind(null,{errorCode:i.INVALID_DO_ON_FREE_HARDWARE}),function(n,e){const t=e.environment;if(!y({environment:t}))return null;return k(t,null,o.REMOVED,i.REMOVED_FREE_HARDWARE)}.bind(null,{errorCode:i.REMOVED_FREE_HARDWARE}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"9a44990c-1aa1-4c7d-baf8-c4099011741c"})&&!A({environment:t,environmentPackage:"ws-python-37-cuda"}))return null;return k(t,null,o.REMOVED,i.REMOVED_GPU_NON_CE)}.bind(null,{errorCode:i.REMOVED_GPU_NON_CE}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(f(t))return null;if(!A({environment:t,environmentPackageRegExp:/-gpu$/}))return null;if(S({environment:t,subModel:"V100"}))return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_BETA_GPU_V100);return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_BETA_GPU_V100}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!f(t))return null;if(S({environment:t,subModel:"k80"}))return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_GPU_K80);return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_GPU_K80}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"e4429883-c883-42b6-87a8-f419d64088cd"})&&!A({environment:t,environmentPackage:"ws-python-37"})&&!A({environment:t,environmentPackage:"dsx-python-37"}))return null;return k(t,null,o.REMOVED,i.REMOVED_PYTHON_NON_CE)}.bind(null,{errorCode:i.REMOVED_PYTHON_NON_CE}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"9447fa8b-2051-4d24-9eef-5acb0e3c59f8"})&&!A({environment:t,environmentPackage:"ws-python-37-premium"})&&!A({environment:t,environmentPackage:"dsx-python-37-premium"}))return null;return k(t,null,o.REMOVED,i.REMOVED_PREMIUM_PYTHON_NON_CE)}.bind(null,{errorCode:i.REMOVED_PREMIUM_PYTHON_NON_CE}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!(n=>{var e,t;return Boolean(null===(t=null===(e=null==n?void 0:n.entity)||void 0===e?void 0:e.environment)||void 0===t?void 0:t.location)})(t))return null;return k(t,null,o.REMOVED,i.REMOVED_SATELLITE_NOTEBOOK)}.bind(null,{errorCode:i.REMOVED_SATELLITE_NOTEBOOK}),function(n,e){const t=e.compute;if(!b({compute:t,type:"spark"}))return null;return k(null,t,o.REMOVED,i.REMOVED_SPARK_AS_A_SERVICE)}.bind(null,{errorCode:i.REMOVED_SPARK_AS_A_SERVICE}),function(n,e){const t=e.environment,r=e.entitlements;let u=!0;b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"})?function(n){const e=n.environment,t=n.entitlements,r=E()(t,"properties.environments.notebooks.max_cpu_cores"),o=Number(v(e));return!o||!r||o<=r}({environment:t,entitlements:r})||(u=!1):b({environment:t,type:"default_spark"})?(function(n){const e=n.environment,t=n.entitlements,r=E()(t,"properties.environments.spark.max_driver_cpu_cores"),o=Number(D(e));return!o||!r||o<=r}({environment:t,entitlements:r})||(u=!1),function(n){const e=n.environment,t=n.entitlements,r=E()(t,"properties.environments.spark.max_executors"),o=Number(P(e));return!o||!r||o<=r}({environment:t,entitlements:r})||(u=!1),function(n){const e=n.environment,t=n.entitlements,r=E()(t,"properties.environments.spark.max_executor_cpu_cores"),o=Number(p(e));return!o||!r||o<=r}({environment:t,entitlements:r})||(u=!1)):b({environment:t,type:"rstudio"})&&(function(n){const e=n.environment,t=n.entitlements,r=E()(t,"properties.environments.rstudio.max_cpu_cores"),o=Number(v(e));return!o||!r||o<=r}({environment:t,entitlements:r})||(u=!1));if(u)return null;return k(t,null,o.REMOVED,i.REMOVED_HARDWARE_CONFIG_FROM_PLAN)}.bind(null,{errorCode:i.REMOVED_HARDWARE_CONFIG_FROM_PLAN}),T.bind(null,{errorCode:i.REMOVED_COMPUTE_FROM_PLAN}),function(n,e){const t=e.environment,r=E()(t,"entity.environment.compute_specification"),o=e.entitlements;if(!b({environment:t,type:"remote_spark"}))return null;return T(0,{compute:r,entitlements:o})}.bind(null,{errorCode:i.REMOVED_IAE_FROM_PLAN}),function(n,e){const t=e.environment;if(K({environment:t,softwareSpecId:"2b7961e2-e3b1-5a8c-a491-482c8368839a"}))return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_39);return null}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_39}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.8"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_38)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_38}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"96e60351-99d4-5a1c-9cc0-473ac1b5a864"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NLP_BETA)}.bind(null,{errorCode:i.REMOVED_NLP_BETA}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.7"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_37)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_37}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.6"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_36)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_36}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.5"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_35)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_35}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"2.7"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_27)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_27}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackage:"dsx-python-35-premium"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_35_DO)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_35_DO}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackage:"dsx-python-36-premium"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_36_DO)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_36_DO}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackage:"dsx-python-27"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_27)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_27}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackage:"dsx-python-35"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_35)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_35}),function(n,e){const t=e.environment;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.6"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_36)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_36}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!A({environment:t,environmentPackage:"dsx-r"}))return null;if(!V({environment:t,language:"r"}))return null;if(C({environment:t,version:"3.4"}))return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R34);return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R34}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"2.3"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_23_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_SPARK_23_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"1b70aec3-ab34-4b87-8aa0-a4a3c8296a36"})&&!A({environment:t,environmentPackage:"dsx-r36"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R36_ON_PYTHON37)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R36_ON_PYTHON37}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"2.4"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_24_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_SPARK_24_ON_CLOUD}),function(n,e){const t=e.environment;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.7"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_37)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_37}),function(n,e){const t=e.environment;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.8"}))return null;if(K({environment:t,softwareSpecId:"96e60351-99d4-5a1c-9cc0-473ac1b5a864"}))return null;return k(t,null,o.REMOVED,i.REMOVED_PYTHON_38_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_PYTHON_38_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"3.0"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_30_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_SPARK_30_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!K({environment:t,softwareSpecId:"41c247d3-45f8-5a71-b065-8580229facf0"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R36_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R36_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!K({environment:t,softwareSpecId:"1c9e5454-f216-59dd-a20e-474a5cdf5988"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_R36_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_SPARK_R36_ON_CLOUD}),function(n,e){const t=e.environment;if(!b({environment:t,type:"notebook"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.9"}))return null;return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_39_ON_CLOUD)}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_39_ON_CLOUD}),function(n,e){const t=e.environment,r=e.entitlements,u=null==r?void 0:r.bss_account_name;let _=!0;if("string"!=typeof u||u.trim().toLowerCase()!=="WatsonX Challenge".trim().toLowerCase())return null;b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"})?(function(n){const e=n.environment,t=n.limit,r=Number(v(e));return r&&t&&r>t}({environment:t,limit:2})&&(_=!1),N(t)&&(_=!1)):b({environment:t,type:"default_spark"})&&(function(n){const e=n.environment,t=n.limit,r=Number(D(e));return r&&t&&r>t}({environment:t,limit:1})&&(_=!1),Y({environment:t,limit:1})&&(_=!1),Y({environment:t,limit:1})&&(_=!1),function(n){const e=n.environment,t=n.limit,r=Number(P(e));return r&&t&&r>t}({environment:t,limit:2})&&(_=!1));if(_)return null;return k(t,null,o.INVALID,i.INVALID_FOR_WATONSX_CHALLENGE)}.bind(null,{errorCode:i.INVALID_FOR_WATONSX_CHALLENGE});(function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"scala"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SCALA)}).bind(null,{errorCode:i.REMOVED_SCALA}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R34),_=M({environment:t});if(_){if(_===u.JUPYTER_R)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"rstudio"}))&&V({environment:t,language:"r"})&&C({environment:t,version:"3.4"}))return r;return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R34}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_35),_=M({environment:t});if(_){if(_===u.JUPYTER_PY35||_===u.JUPYTER_PY35_GPU)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.5"}))return r;return null}.bind(null,{errroCode:i.REMOVED_NOTEBOOK_PYTHON_35}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_36),_=M({environment:t});if(_){if(_===u.JUPYTER_PY36||_===u.JUPYTER_LAB)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.6"}))return r;return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_36}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_PYTHON_NON_CE),_=M({environment:t});if(_){if(_===u.JUPYTER_PY37_LEGACY||_===u.LAB_PY37_LEGACY)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.7"})&&(K({environment:t,softwareSpecId:"e4429883-c883-42b6-87a8-f419d64088cd"})||K({environment:t,softwareSpecId:"9a44990c-1aa1-4c7d-baf8-c4099011741c"})))return r;return null}.bind(null,{errorCode:i.REMOVED_PYTHON_NON_CE}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_37),_=M({environment:t});if(_){if(_===u.JUPYTER_PY37||_===u.JUYPTER_LAB_PY37)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.7"})&&K({environment:t,softwareSpecId:"c2057dd4-f42c-5f77-a02f-72bdbd3282c9"}))return r;return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_37}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_38),_=M({environment:t});if(_){if(_===u.JUPYTER_LAB_PY38||_===u.JUPYTER_PY38)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.8"})&&(K({environment:t,softwareSpecId:"ab9e1b80-f2ce-592c-a7d2-4f2344f77194"})||K({environment:t,softwareSpecId:"5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e"})))return r;return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_38}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_PYTHON_39),_=M({environment:t});if(_){if(_===u.JUPYTER_LAB_PY39||_===u.JUPYTER_PY39||_===u.JUPYTER_PY39_GPU||_===u.JUPYTER_LAB_PY39_GPU)return r}else if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"}))&&V({environment:t,language:"python"})&&C({environment:t,version:"3.9"})&&(K({environment:t,softwareSpecId:"12b83a17-24d8-5082-900f-0ab31fbfd3cb"})||K({environment:t,softwareSpecId:"26215f05-08c3-5a41-a1b0-da66306ce658"})))return r;return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_PYTHON_39}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"2.3"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_23)}.bind(null,{errorCode:i.REMOVED_SPARK_23}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"2.4"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_24)}.bind(null,{errorCode:i.REMOVED_SPARK_24}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.7"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_37)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_37}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.8"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_38)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_38}),function(n,e){const t=e.environment,r=e.cpuArchitecture,_=M({environment:t});if(_&&_===u.JUPYTER_R36&&"ppc64"!==r)return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R36_ON_PYTHON38);return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R36_ON_PYTHON38}),function(n,e){const t=e.environment,r=M({environment:t});if(r&&r===u.JUPYTER_R36_PY39)return k(t,null,o.REMOVED,i.REMOVED_NOTEBOOK_R36);return null}.bind(null,{errorCode:i.REMOVED_NOTEBOOK_R36}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"3.0"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_30)}.bind(null,{errorCode:i.REMOVED_SPARK_30}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!I({environment:t,version:"3.2"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_32)}.bind(null,{errorCode:i.REMOVED_SPARK_32}),function(n,e){const t=e.environment,r=M({environment:t});if((b({environment:t,type:"notebook"})||b({environment:t,type:"jupyterlab"})||b({environment:t,type:"rstudio"}))&&!r)return k(t,null,o.INVALID,i.MISSING_RUNTIME_DEF_ID_CPD);return null}.bind(null,{errorCode:i.MISSING_RUNTIME_DEF_ID_CPD}),function(n,e){const t=e.environment,r=k(t,null,o.REMOVED,i.REMOVED_RSTUDIO_R36),_=M({environment:t});if(_){if(_===u.RSTUDIO)return r}else if(b({environment:t,type:"rstudio"})&&V({environment:t,language:"r"})&&C({environment:t,version:"3.6"}))return r;return null}.bind(null,{errorCode:i.REMOVED_RSTUDIO_R36}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"r"}))return null;if(!C({environment:t,version:"3.6"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_R36)}.bind(null,{errorCode:i.REMOVED_SPARK_R36}),function(n,e){const t=e.environment;if(!b({environment:t,type:"default_spark"}))return null;if(!V({environment:t,language:"python"}))return null;if(!C({environment:t,version:"3.9"}))return null;return k(t,null,o.REMOVED,i.REMOVED_SPARK_PYTHON_39)}.bind(null,{errorCode:i.REMOVED_SPARK_PYTHON_39})}).call(this,t("8oxB"))},ycre:function(n,e,t){var r=t("711d")("length");n.exports=r}}]);