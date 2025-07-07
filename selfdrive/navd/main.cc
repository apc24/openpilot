#include <csignal>
#include <sys/resource.h>

#include <QApplication>
#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/navd/map_renderer.h"
#include "system/hardware/hw.h"

int main(int argc, char *argv[]) {
  Hardware::config_cpu_rendering(true);

  qInstallMessageHandler(swagLogMessageHandler);
  setpriority(PRIO_PROCESS, 0, -20);
  int ret = util::set_core_affinity({0, 1, 2, 3});
  assert(ret == 0);

  QApplication app(argc, argv);
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  // DUAL MAP SERVICES ARCHITECTURE:
  // 1. MapTiler: Used for map display/rendering (this process)
  // 2. Mapbox: Used for navigation routing (handled by navd.py process)
  
  // MapTiler for map display and tile rendering
  MapRenderer * m = new MapRenderer(get_maptiler_settings());
  assert(m);

  // Note: Mapbox routing is handled separately by navd.py process
  // Both services coexist to provide complete navigation functionality

  return app.exec();
}
