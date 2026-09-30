"""Native Qt presentation: restrained colours, readable inputs and boundary sketch."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


STYLESHEET = """
QMainWindow { background: #f1f5f9; }
QWidget { font-family: 'PingFang SC'; font-size: 13px; color: #25394d; }
QLabel#AppTitle { font-size: 25px; font-weight: 600; color: #163047; }
QLabel#Badge { color: #126a67; background: #dff2ec; border-radius: 10px; padding: 5px 12px; font-size: 12px; }
QLabel#WorkspacePath { color: #62798e; font-size: 11px; padding: 0 2px 6px; }
QLabel#Muted { color: #698095; font-size: 12px; }
QWidget#Sidebar { background: #142c41; border-radius: 14px; }
QWidget#Sidebar QLabel { color: #c2d1de; font-size: 12px; }
QWidget#Sidebar QLabel#SidebarTitle { color: #ffffff; font-size: 15px; font-weight: 600; }
QWidget#Sidebar QListWidget { background: transparent; border: none; color: #d4e1eb; outline: none; }
QWidget#Sidebar QListWidget::item { padding: 12px 10px; border-radius: 8px; margin: 3px 0; }
QWidget#Sidebar QListWidget::item:hover { background: #213f56; }
QWidget#Sidebar QListWidget::item:selected { background: #227a79; color: white; }
QWidget#Sidebar QPushButton { background: #233f55; color: #eff5f9; border: 1px solid #385469; }
QWidget#Sidebar QPushButton:hover { background: #30556f; }
QWidget#Sidebar QPushButton:disabled { color: #7890a2; background: #20364a; border-color: #294256; }
QTabWidget::pane { background: white; border: 1px solid #dfe7ee; border-radius: 10px; top: -1px; }
QTabBar::tab { background: transparent; color: #718597; padding: 11px 20px; margin: 0 4px 7px 0; border-radius: 7px; }
QTabBar::tab:selected { background: #e1f0ed; color: #116a67; font-weight: 600; }
QTabBar::tab:hover:!selected { background: #e8eef4; color: #304d64; }
QTabWidget#ConfigTabs::pane { border: none; background: white; }
QTabWidget#ConfigTabs QTabBar::tab { padding: 9px 15px; font-size: 12px; }
QGroupBox { background: #f8fafc; border: 1px solid #e5ebf1; border-radius: 10px; margin-top: 17px; padding: 20px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 15px; color: #486176; }
QLineEdit, QComboBox { background: white; border: 1px solid #d4dfe8; border-radius: 6px; padding: 7px 9px; min-height: 17px; selection-background-color: #c5e9e1; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #248a81; background: #fcfefd; }
QComboBox::drop-down { border: none; width: 25px; }
QComboBox::down-arrow { image: url('__DOWN_ARROW__'); width: 12px; height: 8px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #def0ec; selection-color: #164e4b; padding: 5px; }
QCheckBox { spacing: 9px; padding: 5px 0; font-weight: normal; }
QCheckBox::indicator { width: 16px; height: 16px; }
QPushButton { background: white; border: 1px solid #cfdbe5; border-radius: 7px; padding: 8px 13px; color: #345369; font-weight: 500; }
QPushButton:hover { background: #eef6f5; border-color: #87b6b0; color: #126b65; }
QPushButton:pressed { background: #dceee9; }
QPushButton:disabled { color: #9baab8; background: #f0f3f6; border-color: #e0e6eb; }
QPushButton#Primary { background: #177d75; color: white; border-color: #177d75; font-weight: 600; padding: 10px 22px; }
QPushButton#Primary:hover { background: #126b65; }
QPushButton#Primary:disabled { background: #a3c5c0; border-color: #a3c5c0; }
QPushButton#Cancel { color: #a05339; border-color: #dec7bb; background: #fff9f5; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: white; }
QScrollBar:vertical { width: 9px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: #ccd7e0; border-radius: 3px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QPlainTextEdit { background: #f8fafc; border: 1px solid #e1e8ef; border-radius: 8px; padding: 8px; font-size: 12px; }
QPlainTextEdit#Preflight { background: #eef7f4; border-color: #d5e9e1; color: #245a4f; }
QLabel#RunStatus { font-size: 18px; font-weight: 600; color: #163b4e; padding: 7px 0; }
QLabel#Metrics { background: #eef4f8; border: 1px solid #dde7ef; border-radius: 9px; padding: 14px; font-size: 13px; }
QProgressBar { border: none; background: #e5ecef; border-radius: 5px; height: 10px; text-align: center; font-size: 10px; color: #2e5159; }
QProgressBar::chunk { background: #279588; border-radius: 5px; }
QTableWidget { background: white; alternate-background-color: #f6f9fb; border: 1px solid #e0e8ef; border-radius: 8px; gridline-color: #edf1f5; selection-background-color: #e0f1eb; selection-color: #135d54; }
QHeaderView::section { background: #edf3f7; color: #546f83; border: none; padding: 10px 8px; font-size: 12px; font-weight: 600; }
QTableWidget::item { padding: 9px 6px; border-bottom: 1px solid #edf1f5; }
QTableWidget::item:selected { background: #e0f1eb; color: #135d54; }
QSplitter::handle { background: transparent; width: 16px; }
QStatusBar { color: #637e90; font-size: 11px; }
QToolBar { background: white; border: none; spacing: 6px; }
QToolTip { color: #eff5fa; background: #203b50; border: none; padding: 6px; }
""".replace('__DOWN_ARROW__', str(Path(__file__).with_name('chevron-down.svg')))


class BoundarySketch(QWidget):
    """Boundary/force diagram, explicitly schematic rather than a computed field."""
    def __init__(self) -> None:
        super().__init__()
        self.config = None
        self.setMinimumHeight(165)
        self.setMinimumWidth(260)

    def set_config(self, config) -> None:
        self.config = config
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#f8fafc"))
        p.setFont(QFont("PingFang SC", 10))
        p.setPen(QColor("#6a8295"))
        p.drawText(QRectF(0, 2, self.width(), 24), Qt.AlignCenter, "边界与驱动示意 · 非网格比例")
        if self.config is None:
            p.end(); return
        c = self.config
        rect = QRectF(62, 59, max(80, self.width()-124), max(55, self.height()-117))
        if c.case_type == "lid_driven_cavity":
            side = min(rect.width(), rect.height())
            rect = QRectF((self.width()-side)/2, 59, side, side)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#e2f1f6"))
        p.drawRoundedRect(rect, 3, 3)
        p.setPen(QPen(QColor("#c8e0e8"), 1, Qt.DotLine))
        for i in range(1, 8):
            x = rect.left()+rect.width()*i/8
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for i in range(1, 5):
            y = rect.top()+rect.height()*i/5
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        labels = {}
        for side, bc in c.boundaries.items():
            periodic = bc.type == "periodic"
            moving = "moving" in bc.type or abs(bc.ux)+abs(bc.uy)>0
            pressure = bc.type == "pressure_zou_he"
            labels[side] = ("周期" if periodic else f"ρ={bc.rho:.5g}" if pressure else
                            f"u={bc.ux:g}" if moving else "静壁" if "bounce" in bc.type else "开放边界")
            p.setPen(QPen(QColor("#2588a0" if periodic else "#1d897d" if moving else "#537489"),
                           2, Qt.DashLine if periodic else Qt.SolidLine))
            ends = {"top":(rect.topLeft(),rect.topRight()),"bottom":(rect.bottomLeft(),rect.bottomRight()),
                    "left":(rect.topLeft(),rect.bottomLeft()),"right":(rect.topRight(),rect.bottomRight())}
            p.drawLine(*ends[side])
        p.setPen(QColor("#577286"))
        p.drawText(QRectF(rect.left()-20,rect.top()-29,rect.width()+40,22),Qt.AlignCenter,labels['top'])
        p.drawText(QRectF(rect.left()-20,rect.bottom()+7,rect.width()+40,22),Qt.AlignCenter,labels['bottom'])
        p.drawText(QRectF(0,rect.center().y()-22,60,44),Qt.AlignCenter|Qt.TextWordWrap,labels['left'])
        p.drawText(QRectF(rect.right()+4,rect.center().y()-22,self.width()-rect.right()-4,44),Qt.AlignCenter|Qt.TextWordWrap,labels['right'])
        ax, ay = c.flow.body_force_x, c.flow.body_force_y
        if ax or ay:
            import math
            length=math.hypot(ax,ay)
            direction=QPointF(ax/length,-ay/length)
            start=rect.center()-direction*20
            end=rect.center()+direction*20
            self._arrow(p,start,end)
        elif c.boundaries['top'].ux:
            x=rect.center().x();y=rect.top()-5
            sign=1 if c.boundaries['top'].ux>0 else -1
            self._arrow(p,QPointF(x-22*sign,y),QPointF(x+22*sign,y))
        if c.grid.obstacle_type != "none":
            p.setPen(Qt.NoPen);p.setBrush(QColor("#425a6d"))
            center=QPointF(rect.left()+c.grid.obstacle_x*rect.width(),rect.bottom()-c.grid.obstacle_y*rect.height())
            if c.grid.obstacle_type=='circle':
                radius=c.grid.obstacle_radius*min(rect.width(),rect.height())
                p.drawEllipse(center,radius,radius)
            else:
                w=c.grid.obstacle_width*rect.width();h=c.grid.obstacle_height*rect.height()
                p.drawRect(QRectF(center.x()-w/2,center.y()-h/2,w,h))
        p.setPen(QColor("#6a8295"))
        p.drawText(QRectF(0,self.height()-22,self.width(),20),Qt.AlignCenter,f"{c.grid.NX} × {c.grid.NY} 流体格点框架 · 箭头仅表示驱动")
        p.end()

    @staticmethod
    def _arrow(p: QPainter, start: QPointF, end: QPointF) -> None:
        import math
        p.setPen(QPen(QColor("#178777"),2.5))
        p.drawLine(start,end)
        angle=math.atan2(end.y()-start.y(),end.x()-start.x())
        a=end-QPointF(8*math.cos(angle-.45),8*math.sin(angle-.45))
        b=end-QPointF(8*math.cos(angle+.45),8*math.sin(angle+.45))
        p.setBrush(QColor("#178777"));p.drawPolygon(QPolygonF([end,a,b]))
